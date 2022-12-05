import ray


def map_to_remote(scheduler_fn, inputs, NO_workers):
    """Input is mapped in batches of NO_workers, to avoid memory issues"""

    # print(f"Map_to_remote with {len(inputs)} input values", flush=True)

    results = []

    current_index = 0
    finished = False

    while True:

        object_refs = []

        for i in range(NO_workers):

            # print(f"Scheduling {current_index}", flush=True)

            if current_index == len(inputs):
                finished = True
                break

            ref = scheduler_fn(inputs[current_index])
            object_refs.append(ref)

            current_index += 1

        # print("Waiting for results of this batch...", flush=True)

        results.extend(
            ray.get(object_refs)
        )  # wait for everything to finish before starting the next batch

        if finished:
            break

    # print("Finished.", flush=True)

    return results
