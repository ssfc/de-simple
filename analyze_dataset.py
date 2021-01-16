with open("datasets/train.txt", "r", encoding='UTF-8') as f:
    data = f.readlines()

facts = []
for line in data:
    elements = line.strip().split("\t")

    head_id = self.getEntID(elements[0])
    rel_id = self.getRelID(elements[1])
    tail_id = self.getEntID(elements[2])
    timestamp = elements[3]

    facts.append([head_id, rel_id, tail_id, timestamp])


