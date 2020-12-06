"""
Mostly copied from the test scripts in
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/
"""
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import time


def main(control="gym-like"):
    env = KukaGymEnv(renders=True, isDiscrete=False, maxSteps=10000000)
    motorsIds = []
    if control == "gym-like":
        dv = 1
        motorsIds.append(env._p.addUserDebugParameter("posX", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("posY", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("yaw", -dv, dv, 0))
    else:
        dv = 0.01
        motorsIds.append(env._p.addUserDebugParameter("posX", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("posY", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("posZ", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("yaw", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("fingerAngle", 0, 0.3, 0.3))

    done = False
    i = 0
    while not done:
        time.sleep(0.01)
        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))
        if control == "gym-like":
            state, reward, done, info = env.step(action)
        else:
            state, reward, done, info = env.step2(action)
        obs = env.getExtendedObservation()
        print(i)
        print(f"Action: {action}")
        print(f"Observation: {obs}")
        i += 1


if __name__ == "__main__":
    control = "gym-like"
    main(control)
