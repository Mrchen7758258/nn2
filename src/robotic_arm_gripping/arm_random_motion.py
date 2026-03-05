import mujoco
import mujoco.viewer
import numpy as np
import time
import os

MODEL_PATH = "arm_model.xml"

def main():
    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在：{MODEL_PATH}")
        return
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 执行器ID
    act_ids = [
        model.actuator("shoulder").id,
        model.actuator("elbow").id,
        model.actuator("left").id,
        model.actuator("right").id
    ]

    # 可视化设置
    viewer = mujoco.viewer.launch(model, data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 30
    viewer.cam.elevation = -15
    viewer.cam.lookat = [0.2, 0, 0.4]

    print("===== 机械臂随机运动测试 =====")
    print("所有关节会随机摆动，夹爪随机开合 | ESC退出")

    step = 0
    # 每50步随机更换一次力矩
    while viewer.is_running():
        if step % 50 == 0:
            # 随机生成力矩（范围：-2~2，匹配模型ctrlrange）
            random_torques = np.random.uniform(-2, 2, size=4)
            for i, act_id in enumerate(act_ids):
                data.ctrl[act_id] = random_torques[i]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
        step += 1

    viewer.close()

if __name__ == "__main__":
    main()