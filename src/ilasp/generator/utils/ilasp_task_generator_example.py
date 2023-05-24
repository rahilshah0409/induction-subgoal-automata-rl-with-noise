from ilasp.ilasp_common import OBS_STR, generate_injected_statement


def generate_examples(goal_examples, dend_examples, inc_examples):
    is_rejecting = len(dend_examples) > 0  # whether the rejecting state is used

    examples, ids = _generate_goal_examples(goal_examples, is_rejecting)
    dend_examples_str, n_ids = _generate_deadend_examples(dend_examples)
    examples += dend_examples_str
    ids.extend(n_ids)
    inc_examples_str, inc_ids = _generate_incomplete_examples(inc_examples, is_rejecting)
    examples += inc_examples_str + "\n"
    ids.extend(inc_ids)
    examples += _generate_examples_injection(ids) + '\n'
    return examples


def _generate_examples_injection(eg_ids):
    eg_ids_str = ""
    for i in range(len(eg_ids)):
        eg_ids_str += eg_ids[i]
        if i != len(eg_ids) - 1:
            eg_ids_str += ";"
    return generate_injected_statement("example_active(" + eg_ids_str + ").")


def get_longest_example_length(goal_examples, dend_examples, inc_examples):
    max_goal = len(max(goal_examples, key=len)) if len(goal_examples) > 0 else 0
    max_dend = len(max(dend_examples, key=len)) if len(dend_examples) > 0 else 0
    max_inc = len(max(inc_examples, key=len)) if len(inc_examples) > 0 else 0
    return max(max_goal, max_dend, max_inc)


def _generate_goal_examples(examples, is_rejecting):
    example_str = ""
    ids = []
    for i in range(len(examples)):
        # (example_trace, weight) = example
        example = examples[i]
        id = "p{}".format(i)
        ids.append(id)
        weight = 0.7
        if is_rejecting:
            example_str += "#pos(" + id + "@{}, {{accept}}, {{reject}}, {{\n".format(weight)
        else:
            example_str += "#pos(" + id + "@{}, {{accept}}, {{}}, {{\n".format(weight)
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str, ids


def _generate_deadend_examples(examples):
    example_str = ""
    ids = []
    for i in range(len(examples)):
        example = examples[i]
        id = "n{}".format(i)
        ids.append(id)
        weight = 0.7
        example_str += "#pos(" + id + "@{}, {{reject}}, {{accept}}, {{\n".format(weight)
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str, ids


def _generate_incomplete_examples(examples, is_rejecting):
    example_str = ""
    ids = []
    for i in range(len(examples)):
        example = examples[i]
        id = "i{}".format(i)
        ids.append(id)
        weight = 0.7
        if is_rejecting:
            example_str += "#pos(" + id + "@{}, {{}}, {{accept, reject}}, {{\n".format(weight)
        else:
            example_str += "#pos(" + id + "@{}, {{}}, {{accept}}, {{\n".format(weight)
        example_str += _generate_example(example)
        example_str += "}).\n\n"
    return example_str, ids


def _generate_example(example):
    example_str = "    "
    first = True

    for i in range(0, len(example)):
        for symbol in example[i]:
            if not first:
                example_str += " "
            example_str += "%s(\"%s\", %d)." % (OBS_STR, symbol, i)
            first = False

    if len(example) > 0:
        example_str += "\n"

    example_str += "    last(%d).\n" % (len(example) - 1)

    return example_str
