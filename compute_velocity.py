import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque

############################################################
# 1. 基本函式：計算 3D 質心
############################################################
def compute_3d_centroid_simple(u_coords, v_coords, depth_values):
    """
    給定該 instance 的所有像素 (u, v) 以及深度資訊 depth_values，
    直接用 (u, v, depth) 當作三維空間點計算 centroid。
    回傳值: (centroid_x, centroid_y, centroid_z)
    """
    if len(depth_values) == 0:
        return None
    
    cx = np.mean(u_coords)
    cy = np.mean(v_coords)
    cz = np.mean(depth_values)
    return np.array([cx, cy, cz], dtype=np.float32)

def get_instance_centroid(depth_frame, inst_frame, instance_id):
    """
    在單張影像中，根據 instance_id 取得該 instance 的 (u, v, depth)，
    並計算其質心 (centroid)。
    
    depth_frame: (H, W) 的深度資訊 (浮點或雙精度，代表 metric depth)
    inst_frame : (H, W) 的 instance segmentation (int)
    instance_id: 欲追蹤的目標 ID (int)
    
    回傳: 
      centroid_3d: shape (3,) or None
    """
    mask = (inst_frame == instance_id)
    
    # 找出屬於該 instance 的所有 pixel coordinate
    v_coords, u_coords = np.where(mask)  # (row, col) => (y, x)
    if len(u_coords) == 0:
        return None
    
    # 從 depth_frame 中取出對應之深度
    depth_values = depth_frame[v_coords, u_coords]
    
    # 計算簡易 3D centroid (u, v, depth)
    centroid_3d = compute_3d_centroid_simple(u_coords, v_coords, depth_values)
    return centroid_3d


############################################################
# 2. 視覺化輔助函式
############################################################
def overlay_instance_mask_on_frame(frame, inst_frame, instance_id, color=(0,255,0), alpha=0.3):
    """
    將 inst_frame == instance_id 的區域用指定 color 做半透明疊加。
    
    參數:
      frame      : OpenCV BGR 影像 (會在此上疊加顏色)
      inst_frame : (H, W) 的 instance mask
      instance_id: 要疊加的目標 ID
      color      : BGR 顏色
      alpha      : 透明度 (0~1)
    """
    # 建立影像拷貝
    overlay = frame.copy()

    # 建立該 instance 的遮罩
    mask = (inst_frame == instance_id)
    
    # 將 mask 範圍內的像素設成指定 color
    overlay[mask] = color
    
    # 做 alpha 混合
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def overlay_velocity_info_on_frame(
    frame, 
    centroid_current, 
    velocity_z, 
    speed_z,
    average_depth,
    instance_id
):
    """
    將 centroid 與速度資訊 (速度大小及平均深度) 疊加畫在 frame 上。
    frame: (H, W, 3) BGR 影像 (OpenCV)
    centroid_current: shape (3,) -> (cx, cy, cz)
    velocity_z: float -> 速度在 z 軸的分量
    speed_z: float -> 速度大小 (z 軸)
    average_depth: float -> 平均深度 cz
    instance_id: 物件 ID，用於區分/顯示
    
    注意 (cx, cy) 代表 pixel coords，必須轉成 int 才能畫。
    """
    if centroid_current is None:
        return frame
    
    # 將 (cx, cy) 當作影像座標來畫圈
    cx, cy = int(centroid_current[0]), int(centroid_current[1])
    
    # 在 centroid 上畫一個小圓點
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 綠色實心圓
    
    # 在影像上秀出 instance_id、速度大小 (保留 2 位小數)
    text_id = f"ID: {instance_id}"
    cv2.putText(frame, text_id, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                0.3, (255, 255, 255), 1, cv2.LINE_AA)
    
    text_speed = f"Speed Z: {speed_z * 3.6:.2f} km/hr"
    cv2.putText(frame, text_speed, (cx+10, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.3, (255, 0, 0), 1, cv2.LINE_AA)
    
    # 顯示平均深度 (保留 2 位小數)
    text_depth = f"Depth: {average_depth:.2f}m"
    cv2.putText(frame, text_depth, (cx+10, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 
                0.3, (255, 0, 0), 1, cv2.LINE_AA)
    
    return frame


############################################################
# 3. 主流程：逐格讀取影片，計算並可視化
############################################################
def process_video_and_overlay_all_instances(
    video_path,
    depth_dir,
    inst_dir,
    output_path,
    fps=30,
    max_distance=5.0,
    moving_average_window=5
):
    """
    逐格讀取原始影片，同步載入 depth_x.npy, inst_x.npy，
    對於該影格中的「所有 instance ID」計算其與前一幀的速度 (僅基於深度變化)，
    並把結果畫在每格影像上。同時會將前後幀的 instance 區域都疊加在畫面上。
    
    新增功能：
      - 使用移動平均平滑速度計算
      - 在每個 instance 的影像上顯示平均深度
    
    假設:
      - segmentation mask 中的 0 為背景，其餘正整數 ID 為有效物體。
      - 每個 frame i 對應 depth_i.npy, inst_i.npy
    
    參數:
      video_path:       原始影片檔案路徑
      depth_dir :       深度影像資料夾 (內含 depth_0.npy, depth_1.npy, ...)
      inst_dir  :       instance segmentation 資料夾 (內含 inst_0.npy, inst_1.npy, ...)
      output_path:      視覺化後輸出的影片檔路徑
      fps        :      影片的 fps
      max_distance:     太遠的物體 (centroid.z > max_distance) 不計算、不顯示
      moving_average_window: 移動平均的幀數
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return
    
    # 取得影格寬高資訊
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 建立 VideoWriter (假設要輸出 mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 每兩張相鄰影格的時間差 (秒)
    delta_t = 1.0 / fps  
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片總影格數: {frame_count}")

    # 用於記錄上一影格的 {instance_id: centroid_3d}
    prev_centroids = {}
    # 也順便記錄上一影格的 segmentation mask
    inst_frame_prev = None

    # 收集並排序對應檔案路徑
    depth_paths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".npy")])
    inst_paths  = sorted([os.path.join(inst_dir,  f) for f in os.listdir(inst_dir) if f.endswith(".npy")])
    
    # 初始化移動平均的深度歷史記錄
    depth_histories = {}  # {instance_id: deque([cz1, cz2, ..., czN], maxlen=N)}
    
    for i in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            print("影片讀取完畢或發生錯誤，提早結束。")
            break
        
        # --- 讀取對應的 depth_i.npy, inst_i.npy ---
        if i >= len(depth_paths) or i >= len(inst_paths):
            print(f"找不到對應檔案 (index {i})，跳過。")
            out_writer.write(frame)
            continue
        
        depth_path = depth_paths[i]
        inst_path  = inst_paths[i]
        
        if not (os.path.exists(depth_path) and os.path.exists(inst_path)):
            print(f"檔案不存在: {depth_path} 或 {inst_path}，跳過 frame {i}")
            out_writer.write(frame)
            continue
        
        depth_frame = np.load(depth_path, allow_pickle=True)  # shape: (H, W)
        inst_frame  = np.load(inst_path, allow_pickle=True)   # shape: (H, W)

        # 先把「上一影格」的所有 instance mask 疊加在當前畫面上(紅色)，方便觀察「上一幀」位置
        if inst_frame_prev is not None:
            unique_ids_prev = np.unique(inst_frame_prev)
            for uid_prev in unique_ids_prev:
                if uid_prev == 0:  
                    continue
                overlay_instance_mask_on_frame(
                    frame, inst_frame_prev, uid_prev, 
                    color=(0,0,255),   # 紅色
                    alpha=0.3
                )
        
        # 再把「當前影格」的所有 instance mask 疊加在當前畫面上(綠色)
        unique_ids = np.unique(inst_frame)
        for uid in unique_ids:
            if uid == 0:
                continue
            overlay_instance_mask_on_frame(
                frame, inst_frame, uid, 
                color=(0,255,0),   # 綠色
                alpha=0.3
            )

        # 接著計算「當前影格」所有物體的質心
        current_centroids = {}  # {instance_id: centroid_3d}
        for uid in unique_ids:
            if uid == 0:  
                continue
            centroid_3d = get_instance_centroid(depth_frame, inst_frame, uid)
            # 若質心不存在(例如被完全遮擋等狀況)就略過
            if centroid_3d is None:
                continue
            # 若距離超過 max_distance 也略過，不畫也不算
            if centroid_3d[2] > max_distance:
                continue
            
            current_centroids[uid] = centroid_3d
        
        # 更新深度歷史記錄並計算移動平均
        for uid, centroid_current in current_centroids.items():
            cz = centroid_current[2]
            if uid not in depth_histories:
                depth_histories[uid] = deque(maxlen=moving_average_window)
            depth_histories[uid].append(cz)
        
        # 計算並畫出每個 instance 的速度資訊
        for uid, centroid_current in current_centroids.items():
            velocity_z = 0.0
            speed_z = 0.0
            average_depth = 0.0
            
            # 確認該 instance 有足夠的歷史數據進行移動平均
            if uid in depth_histories and len(depth_histories[uid]) == depth_histories[uid].maxlen:
                # 計算移動平均深度
                average_depth = np.mean(depth_histories[uid])
                
                # 如果在上一影格中也出現同樣的 uid，則可計算速度
                if uid in prev_centroids and 'average_depth_prev' in prev_centroids[uid]:
                    average_depth_prev = prev_centroids[uid]['average_depth_prev']
                    velocity_z = (average_depth - average_depth_prev) / delta_t
                    speed_z = float(np.abs(velocity_z))  # 速度大小取絕對值
                
                # 更新前一幀的平均深度
                prev_centroids[uid]['average_depth_prev'] = average_depth
            elif uid in depth_histories:
                # 尚未達到移動平均窗口大小，使用目前可用的平均
                average_depth = np.mean(depth_histories[uid])
            
            # 疊加畫在 frame 上 (此時畫質心與速度及深度)
            overlay_velocity_info_on_frame(
                frame, centroid_current, 
                velocity_z, speed_z, average_depth, uid
            )
        
        # 更新 prev_centroids
        for uid in current_centroids.keys():
            if uid not in prev_centroids:
                prev_centroids[uid] = {}
        
        # 更新 inst_frame_prev
        inst_frame_prev = inst_frame.copy()
        
        # 寫入輸出影片
        out_writer.write(frame)

    cap.release()
    out_writer.release()
    print(f"輸出完成: {output_path}")


############################################################
# 4. 測試入口
############################################################
if __name__ == "__main__":
    # 範例參數（請依實際檔案做調整）
    original_video_path = "Grounded-SAM-2/car3.mp4"  # 原始影片
    depth_dir = "Depth-Anything-V2/metric_depth/vis_depth_car3"  # 有 depth_0.npy, depth_1.npy, ...
    inst_dir  = "Grounded-SAM-2/car3_outputs/mask_data"             # 有 inst_0.npy, inst_1.npy, ...
    output_video_path = "result_overlay_all_instances_car3.mp4"     # 輸出檔案
    
    fps = 23.98           # 若原影片為 25 FPS（請根據實際情況調整）
    max_distance = 80.0 # 超過 60 公尺的物體不計算、不顯示 (自行調整)
    moving_average_window = 10  # 移動平均的幀數（可自行調整）
    
    process_video_and_overlay_all_instances(
        video_path=original_video_path,
        depth_dir=depth_dir,
        inst_dir=inst_dir,
        output_path=output_video_path,
        fps=fps,
        max_distance=max_distance,
        moving_average_window=moving_average_window
    )
