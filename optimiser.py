from pulp import LpContinuous, LpProblem, LpMinimize, PULP_CBC_CMD, LpStatus, LpVariable, value, LpBinary
from math import pi, degrees
import plotly.express as px
import pandas as pd
class Player:
    def __init__(self, shirt_number: str,  initial_radius: float, theta: float, team: str, has_ball: bool):
        self.initial_radius = initial_radius
        self.theta = theta
        self.shirt_number = shirt_number
        self.team = team
        self.has_ball = has_ball

class Optimiser:
    def __init__(self, players: list[Player]):
        self.players = players
        self.defending_players = [p for p in self.players if p.team == 'defending']
        self.attacking_players = [p for p in self.players if p.team == 'attacking' and not p.has_ball]

    def add_variables(self):
        #continuous variables for r and theta of defending players relative to ball player
        self.defending_player_variables = {}
        for p in self.defending_players:
            self.defending_player_variables[p.shirt_number + '_rpos'] = LpVariable(p.shirt_number + '_rpos', cat=LpContinuous, lowBound = 0) 
            self.defending_player_variables[p.shirt_number + '_theta'] = LpVariable(p.shirt_number + '_theta', cat = LpContinuous, lowBound = 0, upBound=2*pi)
        
        # Binary variables: is_attacker_blocked[attacker] = 1 if attacker is blocked by any defender
        self.is_attacker_blocked = {}
        for attacker in self.attacking_players:
            self.is_attacker_blocked[attacker.shirt_number] = LpVariable(f'blocked_{attacker.shirt_number}', cat=LpBinary) 

    def create_model(self):
        self.model = LpProblem("defensive_positions", LpMinimize)

    def add_objective(self):
        #minimize number of blocked attackers and distance moved
        blocked_attackers = sum(self.is_attacker_blocked[a.shirt_number] for a in self.attacking_players)
        # objective function minimises negative number of blocked attackers + r_0 - r where r is the decision variable of defender radius
        self.model += - blocked_attackers + 0.1 * sum(
            defender.initial_radius - self.defending_player_variables[defender.shirt_number + '_rpos']  for defender in self.defending_players
            )

    def solve_model(self):
        self.model.solve(PULP_CBC_CMD(msg=False))
        print("Status:", LpStatus[self.model.status])
    
    def get_objective_value(self):
        """Return the objective value after solving."""
        if self.model.status == 1:  # Optimal
            return value(self.model.objective)
        return None

    def add_constraints(self, tolerance=0.01, max_radius_change = 1, max_theta_change = pi/4):
        """
        Add constraints to model blocking.
        An attacker is blocked if
        1) any defender's theta is within tolerance of the attacker's theta
        AND
        2) their r value (distance from passer) is greater than the defenders r value (i.e the defender is between the origin and attacker radially)
        """
        #big M constants
        theta_M = 2 * pi + 1 
        radius_M = 1000

        for defender in self.defending_players:
            defender_radius_var = self.defending_player_variables[defender.shirt_number + '_rpos']
            defender_theta_var = self.defending_player_variables[defender.shirt_number + '_theta']
            # Constraint that the player cannot move more than max_radius_change units in either direction radially
            self.model += defender_radius_var <=  defender.initial_radius + max_radius_change 
            self.model += defender_radius_var >= defender.initial_radius - max_radius_change
            
            #ToDo see if we need to change the RHS to account for going negative or above 2pi
            self.model += defender_theta_var >= defender.theta - max_theta_change
            self.model += defender_theta_var <= defender.theta + max_theta_change



        # For each attacker-defender pair, create a binary variable indicating if defender blocks attacker
        blocking_vars = {}
        for attacker in self.attacking_players:
            for defender in self.defending_players:
                var_name = f'blocks_{defender.shirt_number}_{attacker.shirt_number}'
                blocking_vars[(defender.shirt_number, attacker.shirt_number)] = \
                    LpVariable(var_name, cat=LpBinary)
                
                defender_theta_var = self.defending_player_variables[defender.shirt_number + '_theta']
                defender_radius_var = self.defending_player_variables[defender.shirt_number + '_rpos']


                
                # Constraint: if blocking_var = 1, then |defender_theta - attacker_theta| <= tolerance
                # Using big-M: defender_theta - attacker_theta <= tolerance + M*(1 - blocking_var)
                self.model += defender_theta_var - attacker.theta <= tolerance + theta_M * (1 - blocking_vars[(defender.shirt_number, attacker.shirt_number)])
                self.model += attacker.theta - defender_theta_var <= tolerance + theta_M * (1 - blocking_vars[(defender.shirt_number, attacker.shirt_number)])

                #big M constraint for the defending player to be closer to the ball than attacking player if blocking
                self.model += defender_radius_var - attacker.initial_radius <= radius_M*(1 - blocking_vars[(defender.shirt_number, attacker.shirt_number)])
        
        # An attacker is blocked if at least one defender blocks them
        for attacker in self.attacking_players:
            blocking_sum = sum(blocking_vars[(d.shirt_number, attacker.shirt_number)] 
                             for d in self.defending_players)
            # Constraint: is_blocked <= blocking_sum (if no defenders block, is_blocked must be 0)
            # Since we minimize unblocked (maximize is_blocked), solver will set is_blocked = 1 
            # when blocking_sum >= 1
            self.model += self.is_attacker_blocked[attacker.shirt_number] <= blocking_sum

    def create_and_solve_model(self):
        self.create_model()
        self.add_variables()
        self.add_objective()
        self.add_constraints()
        self.solve_model()

    def get_optimal_positions(self):
        pos = []
        shirt_numbers = {'_'.join(var_key.split('_')[0:2]) for var_key in self.defending_player_variables}
        for shirt_number in shirt_numbers:
            pos.append(Player(shirt_number
                              , self.defending_player_variables[shirt_number + '_rpos'].varValue
                              ,self.defending_player_variables[shirt_number + '_theta'].varValue
                              , 'defending_after'
                              , False))
                
        return pos

# players = [
#     Player('player_1', 0, 0, 'attacking', True), #origin is attacking player with ball
#     Player('player_2', 4 , 0, 'defending', False), #def player at distance 3
#     Player('player_3', 3 , pi/2, 'defending', False), #def player at distance 3
#     Player('player_4', 2, pi/4, 'attacking', False),
#     Player('player_5', 3, 0, 'attacking', False),
#     Player('player_6', 3, pi, 'attacking', False)

#     ]
# o = Optimiser(players)
# o.create_and_solve_model()
# print("Objective value:", o.get_objective_value())

# df = pd.DataFrame([p.__dict__ for p in players] + [p.__dict__ for p in o.get_optimal_positions()])
# df['degrees'] = df['theta'].apply(lambda t: degrees(t))
# px.scatter_polar(df, r='initial_radius', theta = 'degrees', symbol = 'team', color='team', opacity=0.5, start_angle = 0, direction = 'counterclockwise')
# print(df)