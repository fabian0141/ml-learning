class HashMap:
    def __init__(self, size):
        self.size = size+1
        self.container = [[[] for _ in range(self.size)] for _ in range(self.size)]


    def place(self, pos):
        #if 0 > pos[0] or pos[0] >= self.size or 0 > pos[1] or pos[1] >= self.size or 0 > pos[2] or pos[2] >= self.size:
        #    return

        row = self.container[pos[0]][pos[1]]
        #self.insertPlace(row, pos[2])
        if pos[2] in row:
            return
        row.append(pos[2])

    def insertPlace(self, row, x):
        if not row:
            row.insert(0, x)

        if row[0] == x:
            return
        index = 0
        while index < len(row) and row[index] < x:
            index += 1
            if row[index] == x:
                return
        
        # Insert the value at the correct position
        row.insert(index, x)



    def convToCSC(self):
        return [((x * 2 + 1) / self.size - 1, (y * 2 + 1) / self.size - 1, (z * 2 + 1) / self.size - 1) for x in range(self.size) for y in range(self.size) for z in self.container[x][y]]
