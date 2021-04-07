"""Contains class environment mechanics

"""

class MentalGym():
    """Is a mental gym. More documentation later.

    Maintains an experiment space, an action bank, and an agent pool.

    Each agent maintains its own Experiment Space, while the action
    bank is maintained by the gym and shared amongst agents. Users
    pass a 'action curation' method when they instantiate the gym;
    that runs remotely and curates the action pool. The default method
    baked into the gym drops 'dead-end' branches if they're under a
    threshold for *metric*.

    It maintains the agent pool using Ray. Each agent is maintaining
    its own Experiment Space. 

    It maintains and curates the action bank by reading and writing
    files to the directory in which the agent stores actions.

    It maintains and curates statistics concerning the actions in
    the action bank. These are shared across agents.8

    """
    def __init__(
        self,
        experiment_space_dimensionality:int=2,
        base_action_set_directory:str='.atomic_actions',
        manage_actions: Callable = lambda x: x
    ):
        # This is the dimensionality of the Experiment Space (i.e. a line, a plane, an n-dimensional hypercube)
        self._ndim = experiment_space_dimensionality
        # This needs a wrapper to test for existence of the directory and make it
        self._action_dir = base_action_set_directory
        # There should be a wrapper function to pop out potential kwargs for ray init
        self._ray_session = ray.init()
        # This creates the experiment space (syntax might be off, but this makes a DataFrame with n columns
        # 
        self._action_space = _build_action_space(self)
        # self._experiment_space = ray.put(_make_experiment_space(experiment_space_dimensionality))
        # Build an agent pool
        # This function creates n agents using the same action space 
        self._agent_pool = [RemoteAgent(self) for ]
    def step(self,id:object,x:Iterable[float]):
        """Places a node into the environment.
        
        Parameters
        ----------
        id: Any
            The id of the node to place
        """

def _add_node(experiment_space):
    
def _make_experiment_space(
    dataset: Union[np.array,pd.DataFrame],# Can prob just implement for any array like with columns
    n_dimensions:int=2
)->pd.DataFrame:
    """Build learning environment.

    This function outputs a dataframe with n rows and five columns
    with names ["node_id", "node_type", "location","inputs","outputs"]
    and data types [Object, Object, np.array[float],np.array[int],np.array[int]]

    The node_id is the population identifier for the action placed.
    The node type is a group identifier, currently used to distinguish
    between input (type 0), output (type -1), though could be extended.
    The location is of the form <x_0,...,x_n> where n is the
    dimensionality of the experiment space.
    Inputs is a list of input action node population id's.
    Outputs is a list of output action node population id's.

    The n rows within the table, when an experiment begins contain
    solely the set of the input nodes, representing the columns of
    your dataset. This is represented *visually* in the experiment
    space, as a green node representing a data source.
    All are marked as type 0 and uniformly distributed along the
    first axis of experiment space (i.e. positions from (0,0) to (1,0)).
    The output nodes are also located in the table, with a type of -1,
    bringing the total row count to n+1.    

    Parameters
    ------------------
    n_dimensions
    """
    df = pd.DataFrame()
