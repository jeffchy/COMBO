if __name__ == '__main__':
    with open('out.t', 'r') as f:
        s = f.read()
    for l in s.split('\n'):

        if l.startswith('INFO:root:======== clustering mode '):
            print(l)
        if l.startswith("========== LAYER"):
            print(l)
        if l.startswith('INFO:root:test metrics:'):
            print(l)