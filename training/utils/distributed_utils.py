import ray


def map_to_remote(scheduler_fn, inputs, NO_workers):
    """Can be used to map the items of `inputs` to the `scheduler_fn` using
    `NO_workers` worker processes. This mapping will first process `NO_workers`
    items of the `inputs` list and waits for the results until it processes the
    next batch. This is done to fix some memory issues.

    Args:
        scheduler_fn (Callable): function that takes an element of `inputs` and
        calls some ray remote function and returns the handle. inputs (_type_):
        List of inputs to map. NO_workers (_type_): Number of worker processes
        to use.

    Returns:
        list: List of returns of the remote function.
    """

    results = []

    current_index = 0
    finished = False

    while True:

        object_refs = []

        for i in range(NO_workers):

            if current_index == len(inputs):
                finished = True
                break

            ref = scheduler_fn(inputs[current_index])
            object_refs.append(ref)

            current_index += 1

        results.extend(
            ray.get(object_refs)
        )  # wait for everything to finish before starting the next batch

        if finished:
            break

    return results
