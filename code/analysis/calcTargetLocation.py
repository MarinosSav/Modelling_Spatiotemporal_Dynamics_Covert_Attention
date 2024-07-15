targetLocation = cueLocation.copy()
for t in range(len(targetDirection)):
    if targetDirection[t] == 'CW':
            if targetLocation[t] == 'UP':
                targetLocation[t] = 'RIGHT'  
            elif targetLocation[t] == 'RIGHT':
                targetLocation[t] = 'DOWN'  
            elif targetLocation[t] == 'DOWN':
                targetLocation[t] = 'LEFT'  
            elif targetLocation[t] == 'LEFT':
                targetLocation[t] = 'UP'
    if targetDirection[t] == 'CCW':
            if targetLocation[t] == 'UP':
                targetLocation[t] = 'LEFT'  
            elif targetLocation[t] == 'LEFT':
                targetLocation[t] = 'DOWN'  
            elif targetLocation[t] == 'DOWN':
                targetLocation[t] = 'RIGHT'  
            elif targetLocation[t] == 'RIGHT':
                targetLocation[t] = 'UP'
