def count_jumps(list, start_position=0):
    """
    Calculates the number of jumps required to exit the list, starting from a given position.

    Args:
        jumps: A list of non-negative integers representing jump lengths.
        start_position: The initial starting position (index) in the list.

    Returns:
        The number of jumps, or -1 if the process gets stuck.
    """    
    pos = start_position
    ans = 0
    # visited = set()

    while 0 <= pos < len(list):
        # if pos in visited:
        if list[pos] == 0:
            return -1  # Cycle detected, process is stuck
        
        # visited.add(pos)
        jump_value = list[pos]        
        ans += 1
        pos += jump_value
    
    return ans


if __name__ == "__main__":
    input_line = input()
    parts = input_line.split()
    jumps = list(map(int, parts[:-1])) # all but the last number are the jump values
    start_position = int(parts[-1])  # the last number is the start position

    result = count_jumps(jumps, start_position)
    print(result)