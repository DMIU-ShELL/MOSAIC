from minihack.envs.minigrid import MiniGridHack
from gym.envs.registration import register
from gym_minigrid.envs.multiroom import MultiRoomEnv

register(id="Curriculum-MultiRoom-N2-S4-v0",
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 2, "maxNumRooms": 2, "maxRoomSize": 4},
)

register(id='Curriculum-MultiRoom-N3-S4-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 3, "maxNumRooms": 3, "maxRoomSize": 4},
)

register(id='Curriculum-MultiRoom-N4-S4-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 4, "maxNumRooms": 4, "maxRoomSize": 4},
)

register(id='Curriculum-MultiRoom-N5-S4-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 5, "maxNumRooms": 5, "maxRoomSize": 4},
)

register(id='Curriculum-MultiRoom-N6-S4-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 4},
)

register(id="Curriculum-MultiRoom-N7-S4-v0",
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 7, "maxNumRooms": 7, "maxRoomSize": 4},
)

register(id='Curriculum-MultiRoom-N8-S4-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 8, "maxNumRooms": 8, "maxRoomSize": 4},
)


register(id="Curriculum-MultiRoom-N2-S6-v0",
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 2, "maxNumRooms": 2, "maxRoomSize": 6},
)

register(id='Curriculum-MultiRoom-N3-S6-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 3, "maxNumRooms": 3, "maxRoomSize": 6},
)

register(id='Curriculum-MultiRoom-N4-S6-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 4, "maxNumRooms": 4, "maxRoomSize": 6},
)

register(id='Curriculum-MultiRoom-N5-S6-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 5, "maxNumRooms": 5, "maxRoomSize": 6},
)

register(id='Curriculum-MultiRoom-N6-S6-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 6},
)

register(id="Curriculum-MultiRoom-N7-S6-v0",
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 7, "maxNumRooms": 7, "maxRoomSize": 6},
)

register(id='Curriculum-MultiRoom-N8-S6-v0',
    entry_point='CurriculumMinigrid.curriculumMultiRoomEnvMH:MultiRoomEnv',
    kwargs={"minNumRooms": 8, "maxNumRooms": 8, "maxRoomSize": 6},
)


from minihack import MiniHackNavigation, LevelGenerator
from nle.nethack import Command, CompassDirection
import gym


MOVE_AND_KICK_ACTIONS = tuple(
    list(CompassDirection) + [Command.OPEN, Command.KICK]
)

class MiniHackMultiRoomN2S4(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N2-S4-v0", **kwargs
        )
        
class MiniHackMultiRoomN3S4(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*2)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N3-S4-v0", **kwargs
        )

class MiniHackMultiRoomN4S4(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*3)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N4-S4-v0", **kwargs
        )

class MiniHackMultiRoomN5S4(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*4)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N5-S4-v0", **kwargs
        )

class MiniHackMultiRoomN6S4(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*5)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N6-S4-v0", **kwargs
        )

class MiniHackMultiRoomN7S4(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*6)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N7-S4-v0", **kwargs
        )
        
class MiniHackMultiRoomN8S4(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*7)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N8-S4-v0", **kwargs
        )

class MiniHackMultiRoomN2S6(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N2-S6-v0", **kwargs
        )
        
class MiniHackMultiRoomN3S6(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*2)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N3-S6-v0", **kwargs
        )

class MiniHackMultiRoomN4S6(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*3)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N4-S6-v0", **kwargs
        )

class MiniHackMultiRoomN5S6(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*4)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N5-S6-v0", **kwargs
        )

class MiniHackMultiRoomN6S6(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*5)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N6-S6-v0", **kwargs
        )

class MiniHackMultiRoomN7S6(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*6)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N7-S6-v0", **kwargs
        )
        
class MiniHackMultiRoomN8S6(MiniGridHack):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 40*7)
        super().__init__(
            *args, env_name="Curriculum-MultiRoom-N8-S6-v0", **kwargs
        )

register(id="Curriculum-MultiRoomMH-N2-S4-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN2S4"
)

register(id="Curriculum-MultiRoomMH-N3-S4-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN3S4"
)

register(id="Curriculum-MultiRoomMH-N4-S4-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN4S4"
)

register(id="Curriculum-MultiRoomMH-N5-S4-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN5S4"
)

register(id="Curriculum-MultiRoomMH-N6-S4-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN6S4"
)

register(id="Curriculum-MultiRoomMH-N7-S4-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN7S4"
)

register(id="Curriculum-MultiRoomMH-N8-S4-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN8S4"
)



register(id="Curriculum-MultiRoomMH-N2-S6-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN2S6"
)

register(id="Curriculum-MultiRoomMH-N3-S6-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN3S6"
)

register(id="Curriculum-MultiRoomMH-N4-S6-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN4S6"
)

register(id="Curriculum-MultiRoomMH-N5-S6-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN5S6"
)

register(id="Curriculum-MultiRoomMH-N6-S6-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN6S6"
)

register(id="Curriculum-MultiRoomMH-N7-S6-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN7S6"
)

register(id="Curriculum-MultiRoomMH-N8-S6-v0",
    entry_point="CurriculumMinigrid.curriculumMultiRoomEnvMH:MiniHackMultiRoomN8S6"
)