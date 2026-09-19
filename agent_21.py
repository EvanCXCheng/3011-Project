import time
import random
# import timeout_decorator
'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file. If you do so, measure the
    running time of __init__, new_game, update_game, and get_actions yourself
    (for example, with time.perf_counter). This local workaround does not relax
    the one-second limit: it is a hard constraint and will be enforced
    independently during marking.
'''
from agent_baselines import Agent
from collections import deque


''' Thoughts: 
Greedy: defend our SCs, then take others, then take "good" squares (distance to SC and takeable)
'''


class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from agent_baselines.py first.
    
    You can add/override attributes and methods as needed.
    '''

    # @timeout_decorator.timeout(1)
    def __init__(self, agent_name='Give a nickname'):
        super().__init__(agent_name)

        '''Implement your agent here.'''

        self.our_scs = set()
        self.scs = {}

    # @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        '''Implement your agent here.'''

        #Establish supply centers (map of supply center : adjacent locations)
        centers = self.game.map.centers #Player : [supply centers]
        for player in centers:
            if player == power_name:
                for i in centers[player]:
                    self.our_scs.add(i)
            for i in centers[player]:
                #All caps for fleet and army accessible, all lowercase for army only
                self.scs[i] = self.game.map.abut_list(i)

    # @timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy strategy here.
    def update_game(self, all_power_orders):
        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    #Assuming other players attack and defend supply centers optimally, find which can be held and taken
    def check_supply_center(self, sc, armies, fleets, f_armies, f_fleets, used, friendly = False):
        order, potential_friendly, potential_enemy = [], 0, 0

        #If supply center is held
        if friendly and sc in armies:
            potential_friendly = 1
        elif sc in armies:
            potential_enemy = 1

        #Find support and threat
        for adjacent in self.scs[sc]:
            if adjacent in armies or (adjacent.isupper() and adjacent in fleets):
                potential_enemy += 1
            if adjacent not in used and adjacent in f_armies or (adjacent.isupper() and adjacent in fleets):
                potential_friendly += 1

        #Can't be defended
        if friendly and potential_enemy > potential_friendly:
            return [], used
        #Can't be taken
        elif potential_enemy >= potential_friendly:
            return [], used
        
        i = 0
        for adjacent in self.scs[sc]:
            #Have enough support, don't overdo
            if i >= potential_enemy:
                break
            type_ = None

            #Valid army or fleet
            if adjacent in f_armies and adjacent not in used:
                type_ = 'A'
            elif adjacent.isupper() and adjacent in f_fleets and adjacent not in used:
                type_ = 'F'

            if type_:
                used.add(adjacent)
                #First attack has different notation, rest will be support, holding will be done already
                if i == 0 and not friendly:
                    order.append(f"{type_} {adjacent} - {sc}")
                    first = (adjacent, type_)
                else:
                    #Support notation is different for hold and attack
                    if friendly:
                        order.append(f"{type_} {adjacent} S A {sc}")
                    else:
                        order.append(f"{type_} {adjacent} S {first[1]} {first[0]} - {sc}")
            i += 1
        return order, used

    # @timeout_decorator.timeout(1)
    def get_actions(self):

        '''Implement your agent here.'''
        orders = []

        #Adjustment (we probably want to do something here)
        if self.game.phase_type == 'A':
            return []

        #Retreating
        if self.game.phase_type == 'R':
            ours = self.game.get_orderable_locations(self.power_name)
            retreat_moves = {unit : moves for unit, moves in self.game.get_all_possible_orders().items() if moves and unit in ours}

            for unit in retreat_moves:
                moves = retreat_moves[unit]
                if len(moves) == 1: #Must disband
                    orders.append(moves[0])
                else:
                    #BFS to find best space to get to an enemy controlled supply center
                    seen = {unit : None}
                    queue = deque()
                    unit_type = moves[0].split()[0]

                    #Start queue with legal moves only to avoid blocked spaces
                    for move in moves:
                        move = move.split()
                        if len(move) == 4: #Retreat order as A somewhere R somewhere, disband is only 3 as A somewhere D
                            queue.append(move[3])
                            seen[move[3]] = unit

                    while queue:
                        on = queue.popleft()

                        if on in self.scs and on not in self.our_scs:
                            #Find adjacent space (legal move)
                            while seen[on] != unit:
                                on = seen[on]

                            orders.append(f"{unit_type} {unit} R {on}")
                            break

                        neighbours = self.game.map.abut_list(on)
                        for neighbour in neighbours:
                            #Fleet only path
                            if unit_type == 'A' and neighbour[1].islower():
                                continue
                            #Army only path
                            if unit_type == 'F' and neighbour.islower():
                                continue

                            if neighbour not in seen:
                                queue.append(neighbour)
                                seen[neighbour] = on
            return orders
        
        position = self.game.get_units()
        friendly_armies, friendly_fleets, enemy_armies, enemy_fleets, used = set(), set(), set(), set(), set()

        #Find all units, lowercase edge is army only, uppercase is armies and fleets
        for power in position:
            mine = False
            if power == self.power_name:
                mine = True
            for unit in position[power]:
                unit = unit.split()
                if unit[0] == 'A' and mine:
                    friendly_armies.add(unit[1])
                elif unit[0] == 'F' and mine:
                    friendly_fleets.add(unit[1])
                elif unit[0] == 'A':
                    enemy_armies.add(unit[1])
                elif unit[0] == 'F':
                    enemy_fleets.add(unit[1])

        #Hold our supply centers first (maybe this is suboptimal)
        for sc in self.our_scs:
            if sc in friendly_armies:
                orders.append(f"A {sc} H")
                used.add(sc)

        #See if supply centers can be held or attacked, and do where possible
        for sc in self.scs:
            if sc in self.our_scs:
                order, used = self.check_supply_center(sc, enemy_armies, enemy_fleets, friendly_armies, friendly_fleets, used, True)
            else:
                order, used = self.check_supply_center(sc, enemy_armies, enemy_fleets, friendly_armies, friendly_fleets, used)
        orders.extend(order)

        #Now need to decide what to do with remaining units
        orders_remaining = self.game.get_all_possible_orders()
        for army in friendly_armies:
            if army not in used:
                orders.append(random.choice(orders_remaining[army]))
                used.add(army)
        for fleet in friendly_fleets:
            if fleet not in used:
                orders.append(random.choice(orders_remaining[fleet]))
                used.add(fleet)

        return orders

        '''
        Return a list of orders. Each order is a string, with specific format. For the format, read the game rule and game engine documentation.
        
        Expected format:
        A LON H                  # Army at LON holds
        F IRI - MAO              # Fleet at IRI moves to MAO (and attack)
        A WAL S F LON            # Army at WAL supports Fleet at LON (and hold)
        F NTH S A EDI - YOR      # Fleet at NTH supports Army at EDI to move to YOR
        F NWG C A NWY - EDI      # Fleet at NWG convoys Army at NWY to EDI
        A NWY - EDI VIA          # Army at NWY moves to EDI via convoy
        A WAL R LON              # Army at WAL retreats to LON
        A LON D                  # Disband Army at LON
        A LON B                  # Build Army at LON
        F EDI B                  # Build Fleet at EDI

        Note: If an invalid order is sent to the engine, it will be accepted but with a result of 'void' (no effect).
        Note: For a 'support' action, two orders are needed, one for the supporter and one for the supportee. (Same for 'convoy')
        Note: For each unit, if no order is given, it will 'hold' by default.

        Useful Functions:
        
        # This is a dict of all the possible orders for each unit at each location (for all powers).
        possible_orders = self.game.get_all_possible_orders()

        # This is a list of all orderable locations for the power you control.
        orderable_locations = self.game.get_orderable_locations(self.power_name)
    
        # Combining these two, you can have the full action space for the power you control.

        # You can re-use the build_map_graphs function in the GreedyAgent to build the connection graph of the map if needed.
        
        '''