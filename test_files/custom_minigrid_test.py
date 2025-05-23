from gym.envs.registration import register

from gym_minigrid.minigrid import (
    COLOR_NAMES,
    Door,
    Goal,
    Grid,
    MiniGridEnv,
    MissionSpace,
    Wall,
)
from gym_minigrid.wrappers import ReseedWrapper, ImgObsWrapper


class MultiRoom:
    def __init__(self, top, size, entryDoorPos, exitDoorPos):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos


class CurriculumMultiRoomEnv(MiniGridEnv):

    """
    ### Description

    This environment has a series of connected rooms with doors that must be
    opened in order to get to the next room. The final room has the green goal
    square the agent must get to. This environment is extremely difficult to
    solve using RL alone. However, by gradually increasing the number of rooms
    and building a curriculum, the environment can be solved.

    ### Mission Space

    "traverse the rooms to get to the goal"

    ### Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    S: size of map SxS.
    N: number of rooms.

    - `MiniGrid-MultiRoom-N2-S4-v0` (two small rooms)
    - `MiniGrid-MultiRoom-N4-S5-v0` (four rooms)
    - `MiniGrid-MultiRoom-N6-v0` (six rooms)

    """

    def __init__(self, minNumRooms, maxNumRooms, maxRoomSize=10, targetNumRooms=2, **kwargs):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.targetNumRooms = targetNumRooms

        self.rooms = []

        mission_space = MissionSpace(
            mission_func=lambda: "traverse the rooms to get to the goal"
        )

        self.size = 25

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=self.maxNumRooms * 20,
            **kwargs
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self.targetNumRooms#self._rand_int(self.minNumRooms, self.maxNumRooms + 1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (self._rand_int(0, width - 2), self._rand_int(0, width - 2))

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = "yellow"#self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"

    def _placeRoom(self, numLeft, roomList, minSz, maxSz, entryDoorWall, entryDoorPos):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = (
                topX + sizeX < room.top[0]
                or room.top[0] + room.size[0] <= topX
                or topY + sizeY < room.top[1]
                or room.top[1] + room.size[1] <= topY
            )

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(MultiRoom((topX, topY), (sizeX, sizeY), entryDoorPos, None))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = {0, 1, 2, 3}
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (topX + sizeX - 1, topY + self._rand_int(1, sizeY - 1))
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY + sizeY - 1)
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (topX, topY + self._rand_int(1, sizeY - 1))
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY)
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos,
            )

            if success:
                break

        return True
    

from gym.envs.registration import register

register(
    id='CurriculumMultiRoomEnv-v0',
    entry_point='custom_minigrid_multiroom_env:CurriculumMultiRoomEnv',
)
    


import gym
from PIL import Image
from gym_minigrid.wrappers import ReseedWrapper, ImgObsWrapper, RGBImgObsWrapper
import argparse

# Function to convert MiniGrid environment observation to RGB image
def minigrid_to_image(observation):
    # Check if the observation is a tuple
    if isinstance(observation, tuple):
        # If observation is a tuple, extract the first element
        img_data = observation[0]['image']
    else:
        # If observation is a dictionary, extract the 'image' key
        img_data = observation['image']
    
    # Convert image data to a PIL Image
    img = Image.fromarray(img_data)
    
    return img

# Function to save MiniGrid environment as PNG image
def save_minigrid_image(env_name, filename, seed):
    # Create MiniGrid environment
    #env = gym.make(env_name)
    env = CurriculumMultiRoomEnv(minNumRooms=2, maxNumRooms=10, maxRoomSize=6, targetNumRooms=env_name)

    env = RGBImgObsWrapper(env)

    #envs_list = [env1, env2, env3]
    #envs_list = [ReseedWrapper(env, (seed,)) for env, seed in zip(envs_list, [129,112,237])]

    
    env = ReseedWrapper(env, (seed,))
    
    # Reset environment to get initial observation
    obs = env.reset()
    
    # Convert observation to RGB image
    img = minigrid_to_image(obs)
    
    # Save image as PNG
    img.save(filename)
    
    # Close environment
    env.close()
    
if __name__ == '__main__':

    
    #env = ReseedWrapper(ImgObsWrapper(CurriculumMultiRoomEnv(minNumRooms=2, maxNumRooms=10, maxRoomSize=4, targetNumRooms=2)), seeds=[840,])
    env = ReseedWrapper(ImgObsWrapper(gym.make('CurriculumMultiRoomEnv', minNumRooms=2, maxNumRooms=10, maxRoomSize=4, targetNumRooms=2)), seeds=[840,])
    obs = env.reset()

    step = 0

    while True:
        ob, rew, done, truncated, info = env.step(env.action_space.sample())
        print(f"step {step} reward {rew} done {done}")
        step+=1
        if done:
            break


    '''# Example usage:
    seed = 840
    for env_name in range(2, 11):
        filename = 'minigrid_images/CurriculumMultiRoomEnv_' + str(env_name) +'Rooms' + str(seed) + '.png'      # Specify the filename for the saved image
        save_minigrid_image(env_name, filename, seed)'''
