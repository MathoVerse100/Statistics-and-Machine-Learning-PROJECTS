def permuter(array):
    length = len(array)
    permutations = []
    for i in range(length**length):
        permutation = [0] * length
        for j in range(length):
            permutation[j] = array[(i // length**j) % length]
        permutations.append(permutation)

    return permutations


def unique_permuter(array):
    permutations = permuter(array)
    unique_permutations = []

    for permutation in permutations:
        if len(set(permutation)) == len(permutation):
            unique_permutations.append(permutation)

    return unique_permutations


letters = ['c', 'a', 't', 'd', 'o']
p = unique_permuter(letters)

for sub_p in p:
    for letter in sub_p:
        print(letter, end="")
    print()
