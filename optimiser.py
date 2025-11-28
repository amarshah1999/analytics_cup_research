from pulp import LpContinuous, LpProblem, LpMinimize, PULP_CBC_CMD, LpStatus, LpVariable, value, LpBinary
from math import pi
class Player:
    def __init__(self, shirt_number: str,  r: float, theta: float, team: str, has_ball: bool):
        self.r = r
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
        self.defensive_positions = {}
        for p in self.defending_players:
            self.defensive_positions[p.shirt_number + '_rpos'] = LpVariable(p.shirt_number + '_rpos', cat=LpContinuous, lowBound = 0) 
            self.defensive_positions[p.shirt_number + '_theta'] = LpVariable(p.shirt_number + '_theta', cat = LpContinuous, lowBound = 0, upBound=2*pi)
        
        # Binary variables: is_attacker_blocked[attacker] = 1 if attacker is blocked by any defender
        self.is_attacker_blocked = {}
        for attacker in self.attacking_players:
            self.is_attacker_blocked[attacker.shirt_number] = LpVariable(f'blocked_{attacker.shirt_number}', cat=LpBinary) 

    def create_model(self):
        self.model = LpProblem("defensive_positions", LpMinimize)

    def add_objective(self):
        # Minimize the number of unblocked passing options
        # Unblocked = total attackers - blocked attackers
        # So we minimize: num_attackers - sum(is_blocked)
        num_attackers = len(self.attacking_players)
        blocked_attackers = sum(self.is_attacker_blocked[a.shirt_number] for a in self.attacking_players)
        self.model += num_attackers - blocked_attackers 

    def solve_model(self):
        self.model.solve(PULP_CBC_CMD(msg=False))
        print("Status:", LpStatus[self.model.status])
    
    def get_objective_value(self):
        """Return the objective value after solving."""
        if self.model.status == 1:  # Optimal
            return value(self.model.objective)
        return None

    def add_constraints(self, tolerance=0.01, max_radius_change = 0.1):
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
            defender_radius_var = self.defensive_positions[defender.shirt_number + '_rpos']
            # Constraint: |new_radius - initial_radius| <= max_radius_change
            # This becomes:
            # new_radius <= initial_radius + max_radius_change
            self.model += defender_radius_var <= defender.r + max_radius_change
            # new_radius >= initial_radius - max_radius_change (but also >= 0)
            self.model += defender_radius_var >= max(0, defender.r - max_radius_change)


        # For each attacker-defender pair, create a binary variable indicating if defender blocks attacker
        blocking_vars = {}
        for attacker in self.attacking_players:
            for defender in self.defending_players:
                var_name = f'blocks_{defender.shirt_number}_{attacker.shirt_number}'
                blocking_vars[(defender.shirt_number, attacker.shirt_number)] = \
                    LpVariable(var_name, cat=LpBinary)
                
                defender_theta_var = self.defensive_positions[defender.shirt_number + '_theta']
                defender_radius_var = self.defensive_positions[defender.shirt_number + '_rpos']


                
                # Constraint: if blocking_var = 1, then |defender_theta - attacker_theta| <= tolerance
                # Using big-M: defender_theta - attacker_theta <= tolerance + M*(1 - blocking_var)
                self.model += defender_theta_var - attacker.theta <= tolerance + theta_M * (1 - blocking_vars[(defender.shirt_number, attacker.shirt_number)])
                self.model += attacker.theta - defender_theta_var <= tolerance + theta_M * (1 - blocking_vars[(defender.shirt_number, attacker.shirt_number)])
                self.model += defender_radius_var - attacker.r <= radius_M*(1 - blocking_vars[(defender.shirt_number, attacker.shirt_number)])
        
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
        return {k: v.varValue for k,v in self.defensive_positions.items()}

players = [
    Player('player_1', 0, 0, 'attacking', True), #origin is attacking player with ball
    Player('player_2', 3 , 0, 'defending', False), #away player at distance 3
    Player('player_3', 1, pi/4, 'attacking', False)]
o = Optimiser(players)
o.create_and_solve_model()
print("Objective value:", o.get_objective_value())
print("Optimal positions:", o.get_optimal_positions())
