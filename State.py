class State:

    def __init__(self, state, parent, number_moved, move, depth, cost, key):
        self.state = state
        self.parent = parent
        self.number_moved = number_moved
        self.move = move
        self.depth = depth
        self.cost = cost
        self.key = key

        if self.state:
            self.map = ''.join(self.hex_number(e) for e in self.state)

    def hex_number(self, number):
        hex = {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "A",
            11: "B",
            12: "C",
            13: "D",
            14: "E",
            15: "F",
            16: "G",
            17: "H",
            18: "I",
            19: "J",
            20: "K",
            21: "L",
            22: "M",
            23: "N",
            24: "O",
            25: "P"
        }
        return hex.get(number)

    def __eq__(self, other):
        return self.map == other.map

    def __lt__(self, other):
        return self.map < other.map