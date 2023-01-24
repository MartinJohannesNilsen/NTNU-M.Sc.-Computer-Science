# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, raiseNotDefined
from game import Directions
import random
import util
from pacman import PacmanRules, GhostRules, GameState

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        n_food = currentGameState.getFood().count()

        # Only consider food and distance to ghosts in the first place, for simplicity

        # If not eat food by action
        if len(newFood.asList()) == n_food:
            score = float("inf")
            for food_position in newFood.asList():
                new_distance = manhattanDistance(food_position, newPos)
                if new_distance < score:
                    score = -new_distance
        else:
            score = 0

        # Ghosts have to negatively impact on the score
        for ghost in newGhostStates:
            score += manhattanDistance(ghost.getPosition(), newPos)

        # return successorGameState.getScore()
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minmax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minmax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minmax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minmax(gameState, agent, depth):
            # return the eval if Win, Lose or it has reached the depth.
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                # maximize for pacman
                return max(minmax(gameState.generateSuccessor(agent, new_state), 1, depth) for new_state in gameState.getLegalActions(agent))
            else:
                # First calculate next_agent and depth
                next_agent = 0 if gameState.getNumAgents() - 1 == agent else agent + 1
                if next_agent == 0:
                    depth += 1
                # minize for ghosts
                return min(minmax(gameState.generateSuccessor(agent, new_state), next_agent, depth) for new_state in gameState.getLegalActions(agent))

        # Run minmax for pacman as agent 0
        action = Directions.STOP
        best = float("-inf")
        for state in gameState.getLegalActions(0):
            helper = minmax(gameState.generateSuccessor(0, state), 1, 0)
            if helper > best:
                action = state
                best = helper

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minmax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minmax action using self.depth and self.evaluationFunction
        """
        def minimize(gameState, agent, depth, alpha, beta):
            val = float("inf")
            next_agent = 0 if gameState.getNumAgents() - 1 == agent else agent + 1
            if next_agent == 0:
                depth += 1
            for state in gameState.getLegalActions(agent):
                val = min(val, alphabetaprune(gameState.generateSuccessor(agent, state), next_agent, depth, alpha, beta))
                beta = min(beta, val)
                if val < alpha:
                    return val
            return val

        def maximize(gameState, agent, depth, alpha, beta):
            val = float("-inf")
            for state in gameState.getLegalActions(agent):
                val = max(val, alphabetaprune(gameState.generateSuccessor(agent, state), 1, depth, alpha, beta))
                alpha = max(alpha, val)
                if val > beta:
                    return val
            return val

        def alphabetaprune(gameState, agent, depth, alpha, beta):
            # return the eval if Win, Lose or it has reached the depth.
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:
                # maximize for pacman
                return maximize(gameState, agent, depth, alpha, beta)
            else:
                # minize for ghosts
                return minimize(gameState, agent, depth, alpha, beta)

        # Run minmax for pacman as agent 0
        action = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")
        best = float("-inf")
        for state in gameState.getLegalActions(0):
            helper = alphabetaprune(gameState.generateSuccessor(0, state), 1, 0, alpha, beta)
            if helper > best:
                action = state
                best = helper
            if best > beta:
                return best
            alpha = max(alpha, best)

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
