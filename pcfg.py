def lispify_pcfg_program(seq):
    if type(seq) is str:
        seq = seq.strip().split()
    seq.append("END")
    queue = []
    new_seq = []
    in_sym_seq = False
    for token in seq:
        if token in ["append", "prepend", "remove_first", "remove_second"]:
            new_seq.append("(")
            new_seq.append(token)
            queue.append(["two-place", 0])
        elif token in ["copy", "reverse", "shift", "echo", "repeat", "swap_first_last"]:
            new_seq.append("(")
            new_seq.append(token)
            queue.append(["one-place", 0])
        elif token == "," or token == "END":
            if in_sym_seq:
                new_seq.append(")")
                in_sym_seq = False
            while len(queue) > 0:
                if queue[-1][0] == "one-place":
                    _ = queue.pop()
                    new_seq.append(")")
                elif queue[-1][0] == "two-place" and queue[-1][1] == 0:
                    queue[-1][1] = 1
                    break
                elif queue[-1][0] == "two-place" and queue[-1][1] == 1:
                    new_seq.append(")")
                    _ = queue.pop()
        else:
            if not in_sym_seq:
                new_seq.append("'")
                new_seq.append("(")
                in_sym_seq = True
            new_seq.append(token)
    assert new_seq.count("(") == new_seq.count(")"), "Number of opening and closing brackets do not match."
    return " ".join(new_seq)

def lispify_pcfg(source_filename, target_filename):
    outputs = []
    with open(source_filename) as f1, open(target_filename) as f2:
        for line1, line2 in zip(f1, f2):
            outputs.append((
                lispify_pcfg_program(line1),
                "( " + line2.strip() + " )"
            ))
    return outputs

if __name__ == "__main__":
    for output in lispify("./pcfg_data/pcfg_source.txt", "./pcfg_data/pcfg_target.txt"):
        print(output)
