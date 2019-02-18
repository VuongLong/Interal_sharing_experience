STATE_SIZE = 2048
ACTION_SIZE = 4 # action size
HISTORY_LENGTH = 1

NUM_EVAL_EPISODES = 100 # number of episodes for evaluation

TASK_TYPE = 'navigation' # no need to change
# keys are scene names, and values are a list of location ids (navigation targets)
TASK_LIST = {
  'bathroom_02'    : [26, 37, 43, 53, 69],
  'bedroom_04'     : [134, 264, 320, 384, 387],
  'kitchen_02'     : [90, 136, 157, 207, 329],
  'living_room_08' : [92, 135, 193, 228, 254]
}
