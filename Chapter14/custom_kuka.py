import os
import random
import numpy as np
import pybullet as p
from pybullet_envs.bullet.kuka import Kuka
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from alp.alp_gmm import ALPGMM


class CustomKuka(Kuka):
    def __init__(self, *args, jp_override=None, **kwargs):
        self.jp_override = jp_override
        super(CustomKuka, self).__init__(*args, **kwargs)

    def reset(self):
        objects = p.loadSDF(
            os.path.join(self.urdfRootPath, "kuka_iiwa/kuka_with_gripper2.sdf")
        )
        self.kukaUid = objects[0]
        p.resetBasePositionAndOrientation(
            self.kukaUid,
            [-0.100000, 0.000000, 0.070000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        )
        self.jointPositions = [
            0.006418,
            0.413184,
            -0.011401,
            -1.589317,
            0.005379,
            1.137684,
            -0.006539,
            0.000048,
            -0.299912,
            0.000000,
            -0.000043,
            0.299960,
            0.000000,
            -0.000200,
        ]
        if self.jp_override:
            for j, v in self.jp_override.items():
                j_ix = int(j) - 1
                if j_ix >= 0 and j_ix <= 13:
                    self.jointPositions[j_ix] = v

        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(
                self.kukaUid,
                jointIndex,
                p.POSITION_CONTROL,
                targetPosition=self.jointPositions[jointIndex],
                force=self.maxForce,
            )

        self.trayUid = p.loadURDF(
            os.path.join(self.urdfRootPath, "tray/tray.urdf"),
            0.640000,
            0.075000,
            -0.190000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        )
        self.endEffectorPos = [0.537, 0.0, 0.5]
        self.endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []


class CustomKukaEnv(KukaGymEnv):
    def __init__(self, env_config={}):
        renders = env_config.get("renders", False)
        isDiscrete = env_config.get("isDiscrete", False)
        maxSteps = env_config.get("maxSteps", 2000)
        self.rnd_obj_x = env_config.get("rnd_obj_x", 1)
        self.rnd_obj_y = env_config.get("rnd_obj_y", 1)
        self.rnd_obj_ang = env_config.get("rnd_obj_ang", 1)
        self.bias_obj_x = env_config.get("bias_obj_x", 0)
        self.bias_obj_y = env_config.get("bias_obj_y", 0)
        self.bias_obj_ang = env_config.get("bias_obj_ang", 0)
        self.jp_override = env_config.get("jp_override")
        super(CustomKukaEnv, self).__init__(
            renders=renders, isDiscrete=isDiscrete, maxSteps=maxSteps
        )

    def reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(
            os.path.join(self._urdfRoot, "table/table.urdf"),
            0.5000000,
            0.00000,
            -0.820000,
            0.000000,
            0.000000,
            0.0,
            1.0,
        )

        xpos = 0.55 + self.bias_obj_x + 0.12 * random.random() * self.rnd_obj_x
        ypos = 0 + self.bias_obj_y + 0.2 * random.random() * self.rnd_obj_y
        ang = (
            3.14 * 0.5
            + self.bias_obj_ang
            + 3.1415925438 * random.random() * self.rnd_obj_ang
        )
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(
            os.path.join(self._urdfRoot, "block.urdf"),
            xpos,
            ypos,
            -0.15,
            orn[0],
            orn[1],
            orn[2],
            orn[3],
        )

        p.setGravity(0, 0, -10)
        self._kuka = CustomKuka(
            jp_override=self.jp_override,
            urdfRootPath=self._urdfRoot,
            timeStep=self._timeStep,
        )
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def step(self, action):
        dz = -0.0005
        if self._isDiscrete:
            dv = 0.005
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            f = 0.3
            realAction = [dx, dy, dz, da, f]
        else:
            dv = 0.005
            dx = action[0] * dv
            dy = action[1] * dv
            da = action[2] * 0.05
            f = 0.3
            realAction = [dx, dy, dz, da, f]
        obs, reward, done, info = self.step2(realAction)
        return obs, reward / 1000, done, info

    def increase_difficulty(
        self,
    ):
        deltas = {"2": 0.1, "4": 0.1}
        original_values = {"2": 0.413184, "4": -1.589317}

        all_at_original_values = True
        for j in deltas:
            if j in self.jp_override:
                d = deltas[j]
                self.jp_override[j] = max(self.jp_override[j] - d, original_values[j])
                print(f"Joint {j}: {self.jp_override[j]}")
                if self.jp_override[j] != original_values[j]:
                    all_at_original_values = False

        self.rnd_obj_x = min(self.rnd_obj_x + 0.05, 1)
        print(f"Obj. randomization multiplier for x: {self.rnd_obj_x}")
        self.rnd_obj_y = min(self.rnd_obj_y + 0.05, 1)
        print(f"Obj. randomization multiplier for y: {self.rnd_obj_y}")
        self.rnd_obj_ang = min(self.rnd_obj_ang + 0.05, 1)
        print(f"Obj. randomization multiplier for angle: {self.rnd_obj_ang}")

        if self.rnd_obj_x == self.rnd_obj_y == self.rnd_obj_ang == 1:
            if all_at_original_values:
                self.bias_obj_x = 0
                self.bias_obj_y = 0
                self.bias_obj_ang = 0
                print("At maximum difficulty!!!")


class ALPKukaEnv(CustomKukaEnv):
    def __init__(self, env_config={}):
        # Parameters, rnd_obj_x, rnd_obj_y, rnd_obj_ang, jp2, jp4, bias_obj_y
        self.in_training = env_config.get("in_training", True)
        self.rnd_obj_x_min = 0
        self.rnd_obj_x_max = 1
        self.rnd_obj_y_min = 0
        self.rnd_obj_y_max = 1
        self.rnd_obj_ang_min = 0
        self.rnd_obj_ang_max = 1
        self.jp2_min = 0.413184
        self.jp2_max = 1.3
        self.jp4_min = -1.589317
        self.jp4_max = -1
        self.bias_obj_y_min = 0
        self.bias_obj_y_max = 0.04
        self.mins = [
            self.rnd_obj_x_min,
            self.rnd_obj_y_min,
            self.rnd_obj_ang_min,
            self.jp2_min,
            self.jp4_min,
            self.bias_obj_y_min,
        ]
        self.maxs = [
            self.rnd_obj_x_max,
            self.rnd_obj_y_max,
            self.rnd_obj_ang_max,
            self.jp2_max,
            self.jp4_max,
            self.bias_obj_y_max,
        ]
        if self.in_training:
            self.alp = ALPGMM(mins=self.mins, maxs=self.maxs, params={"fit_rate": 20})
            self.task = None
            self.last_episode_reward = None
        self.episode_reward = 0
        super(ALPKukaEnv, self).__init__(env_config)

    def reset(self):
        if self.in_training:
            if self.task is not None and self.last_episode_reward is not None:
                self.alp.update(self.task, self.last_episode_reward)
                print(
                    f"Task recorded: \n",
                    f"--rnd_obj_x: {self.rnd_obj_x}\n",
                    f"--rnd_obj_y: {self.rnd_obj_y}\n",
                    f"--rnd_obj_ang: {self.rnd_obj_ang}\n",
                    f"--jp_override_2: {self.jp_override['2']}\n",
                    f"--jp_override_4: {self.jp_override['4']}\n",
                    f"--bias_obj_y: {self.bias_obj_y}\n",
                    f"---reward: {self.last_episode_reward}\n",
                )
            self.task = self.alp.sample_task()
            self.rnd_obj_x = self.task[0]
            self.rnd_obj_y = self.task[1]
            self.rnd_obj_ang = self.task[2]
            self.jp_override = {"2": self.task[3], "4": self.task[4]}
            self.bias_obj_y = self.task[5]
        else:
            self.rnd_obj_x = self.rnd_obj_x_max
            self.rnd_obj_y = self.rnd_obj_y_max
            self.rnd_obj_ang = self.rnd_obj_ang_max
            self.jp_override = {"2": self.jp2_min, "4": self.jp4_min}
            self.bias_obj_y = self.bias_obj_y_min

        return super(ALPKukaEnv, self).reset()

    def step(self, action):
        obs, reward, done, info = super(ALPKukaEnv, self).step(action)
        self.episode_reward += reward
        if done:
            self.last_episode_reward = self.episode_reward
            self.episode_reward = 0
        return obs, reward, done, info
