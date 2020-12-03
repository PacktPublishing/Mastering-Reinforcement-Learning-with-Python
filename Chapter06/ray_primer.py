# These examples are taken from Ray's own documentation at
# https://docs.ray.io/en/latest/index.html

import ray

# Initialize Ray
ray.init()

# Using remote functions
@ray.remote
def remote_function():
    return 1

object_ids = []
for _ in range(4):
    y_id = remote_function.remote()
    object_ids.append(y_id)

@ray.remote
def remote_chain_function(value):
    return value + 1

y1_id = remote_function.remote()
chained_id = remote_chain_function.remote(y1_id)


# Using remote objects
y = 1
object_id = ray.put(y)

# Using remote classes (actors)
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

a = Counter.remote()
obj_id = a.increment.remote()
ray.get(obj_id) == 1

