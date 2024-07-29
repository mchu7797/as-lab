def get_or_gate(x1, x2):
    w1, w2, bias = 0.5, 0.5, -0.2
    result = x1 * w1 + x2 * w2 + bias
    return 1 if result > 0 else 0


if __name__ == '__main__':
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for x1, x2 in inputs:
        y = get_or_gate(x1, x2)
        print(f"({x1}, {x2}) -> {y}")
