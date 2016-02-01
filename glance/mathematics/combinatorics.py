import numpy as np


def comb(n, k):
    return np.math.factorial(n) // np.math.factorial(k) // np.math.factorial(n-k)


def lexicographic_permutations(seq):
    """Algorithm L for The Art of Computer Programming, Volume 4, Fascicle 2: Generating All Tuples and Permutations."""

    def reverse(seq, start, end):
        '''In-place reversion. Start and end denote the indexes of the last and the first element'''
        while start < end:
            seq[start], seq[end] = seq[end], seq[start]
            start += 1
            end -= 1

    def pairwise_reversed_enumerated(L):
        '''(a,b,c,d) -> ((2,c),(3,d)), ((1,b),(2,c)), ((0,a),(1,b))'''
        length = len(L)
        a1 = zip(range(length-1,-1,-1), reversed(L))
        a2 = zip(range(length-1,-1,-1), reversed(L))
        next(a1, None)
        return zip(a1, a2)

    def reversed_enumerated(L):
        '''(a,b,c,d) -> (3,d), (2,c), (1,b), (0,a)'''
        return zip(range(len(L)-1,-1,-1), reversed(L))


    #[Some checks]
    if not seq:
        raise StopIteration

    try:
        seq[0]
    except TypeError:
        raise TypeError("seq must allow random access.")
    #[end checks]

    last = len(seq)-1
    seq = seq[:] #copy the input sequence
    
    yield seq[:] #return the seq itself as a first value

    while True:
        for ((i1,a1),(i2,a2)) in pairwise_reversed_enumerated(seq):
            if a1 < a2:
                for (l,val) in reversed_enumerated(seq):
                    if a1 < val:
                        seq[i1], seq[l] = seq[l], seq[i1]
                        break
                reverse(seq, i2, last)
                yield seq[:] # Change to yield references to get rid of (at worst) |seq|! copy operations.
                break
        else: #this part will only run when there were no 'break' inside the for-loop
            break #at this position there is no a1<a2, so we have reached the finish, so breaking out of the outer loop
