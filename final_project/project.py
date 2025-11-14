import numpy as np

def getDeck():
    full_deck = np.array(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4)
    np.random.shuffle(full_deck)
    assert full_deck.size == 52
    return full_deck

def getValue(cards):
    values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
        '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 10, 'Q': 10, 'K': 10
    }

    total = 0
    aces = 0

    for card in cards:
        if card == 'A':
            aces += 1
        else:
            total += values[card]

    total += aces * 11

    while total > 21 and aces > 0:
        total -= 10  
        aces -= 1

    return total

def initialState(deck):
    player = [deck[0], deck[1]]
    dealer = [deck[2], deck[3]]
    return player, dealer

def playerPlay(player, deck, index):
    while(getValue(player) < 17):
        action = np.random.randint(0, 2)
        #print("Player chose to", "hit" if action == 0 else "stand")
        if(action == 1):
            break
        player += [deck[index]]
        index += 1
        
    return index

def dealerPlay(dealer, deck, index):
    # Based on the rules of blackjack, dealer must go until value reaches at least 17
    while(getValue(dealer) < 17):
        dealer += [deck[index]]
        index += 1
    return index

def isBlackjack(hand):
    # A blackjack is exactly two cards: Ace + any 10-value card
    if len(hand) != 2:
        return False
    hasAce = 'A' in hand
    hasTenValue = any(card in ['10', 'J', 'Q', 'K'] for card in hand)
    return hasAce and hasTenValue


def getResult(player, dealer):
    p = getValue(player)
    d = getValue(dealer)

    pBJ = isBlackjack(player)
    dBJ = isBlackjack(dealer)

    # Player bust
    if p > 21:
        return 0

    # Dealer bust
    if d > 21:
        return 2

    # Both blackjack → push
    if pBJ and dBJ:
        return 1

    # Player blackjack ONLY case that returns 2.5
    if pBJ:
        return 2.5

    # Dealer blackjack
    if dBJ:
        return 0

    # Normal comparisons
    if p > d:
        return 2
    elif p < d:
        return 0
    else:
        return 1
    
if __name__ == "__main__":
    average = 0
    num_trials = 100000
    for i in range(num_trials):
        deck = getDeck()
        player, dealer = initialState(deck)
        index = 4
        index = playerPlay(player, deck, index)
        index = dealerPlay(dealer, deck, index)
        average += getResult(player, dealer)
    average /= num_trials
    print("Random action EV: " + str(average))