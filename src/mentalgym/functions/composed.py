import json
import os
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from torch import Tensor
from torch.nn.modules.container import ModuleDict
from mentalgym.functions import atomic_constants
from mentalgym.types import ExperimentSpace, FunctionBank
from typing import Any, Dict, Iterable, Optional, Union
import torch.nn as nn

class ComposedFunction(nn.Module):
    """Composed of multiple atomic functions.

    This class is used to build, run, and load a composed function.

    The 'id' is a string representing the *name* of the Function.
    The Composed Function will check the FunctionBank to
        see if an object with this id exists. If it does, then
        the Composed Function will attempt to load its weights
        from that location. If it is unable to load weights then
        an error is raised.
    If it *does not see a folder*, then this will build a model
        from a given Experiment Space.

    Parameters
    ----------
    experiment_space: Optional[ExperimentSpace]
        This is an ExperimentSpace object.
    function_bank: Optional[FunctionBank] = None
        This is a FunctionBank object.

    Methods
    -------
    forward(*args, **kwargs):
        The function signature of forward is such that it expects
        to see the inputs it was instantiated with when it is called
        with forward.

    Properties
    ----------

    """
    def __init__(
        self,
        id: str,
        experiment_space: Optional[ExperimentSpace] = None,
        function_bank: Optional[FunctionBank] = None,
        verbose: bool = True
    ):
        super().__init__()
        self._n_inputs = 0
        self.input = {}
        self._net_subspace = None
        self._verbose = verbose
        self._module_dict = None
        self._function_bank = None
        # This property is storing the *locational* indices that the
        #   graph will expect to see the columns at.
        # i.e. {'input_0': 0, 'input_2': 1}
        # This can be unpacked later to appropriately feed information
        #   to the graph when forward is called.
        self.inputs = {}
        # 1) Is this building the net? We will check to see if the
        #   Function's directory exists.
        self._function_dir = os.path.join(
            function_bank._function_bank_directory,
            id
        )
        folder_exists = os.path.isdir(self._function_dir)
        # If the folder does not exist, then we are going to build
        #   the PyTorch graph for this net.

        if not folder_exists:
            # Get a minimal subspace
            self.build_from_space(experiment_space)
            # Then turn that into a graph.
            self.build_forward()
            # And save that graph.
            self.save()
        # If the folder *does* exist, then we are going to load the
        #   graph for this net.
        else:
            self.load()

    def _recusive_init(
        self, 
        id: str
    ):
        """Populates a ModuleDict with Atomic and Composed functions.

        This uses a minimal subset of information to recursively add
        Atomic and Composed functions to the ModuleDict; this is
        equivalent to calling these layers in an __init__ function.

        Parameters
        ----------
        id: str
            This is the *name* of the layer.
        """
        # If this is your first night at Fight Club, you have to fight.
        if self._module_dict is None:
            self._module_dict = ModuleDict()
        # This fetches just the row for this function.
        # The first time this is called it gets the *sink* row.
        data = self._net_subspace.query("id==@id")
        # This, then, gets the list of input nodes for this function.
        inputs = data.input.iloc[0]

        # If the inputs are None, then this ceases recursion, because
        #   this is a *source* node.
        # TODO: Is this necessary? Can it be purged? The section
        #   below where it checks for the row type of source might
        #   be enough.
        if inputs == None:
            return

        # If the inputs are *not* None, then this has input!
        # For every input node to connect...
        for ind in inputs:
            ########################################################
            #       Instantiate the Input Nodes for this Layer     #
            ########################################################
            # Get the row for this input node:
            row = self._net_subspace.query("id==@ind").to_dict(
                orient = 'records'
            )
            # Pull out the ID; we're going to use this ID as the key
            #   in the ModuleDict.
            err_msg = f"""Composed Function Build Error: recursive_init

            When calling for rows I found a set with != 1 record
            ----------------------------------------------------
            id: {ind}

            type of id: {type(ind)}

            internal subspace
            -----------------\n{self._net_subspace}
            """
            assert len(row)==1, err_msg
            # Now that we *know* it's a single row we get that value
            #   as a Series.
            row = row[0]
            # If it's a *source* then we need to increment the
            #   number of inputs and assign this column a position.
            if row['type'] == 'source':
                self.inputs[ind] = self._n_inputs
                self._n_inputs += 1
                continue
            # Extract the ID
            fn_id = row['id']
            # Then pull out the hyperparameters
            fn_parameters = row['hyperparameters']
            # And pull out the Object.
            fn_object = row['object']
            status_message = f"""Build Composed Layer Init:

            ########################################################
            # Added Function Information                           #
            ########################################################

            Function ID:         {fn_id}
            Function Parameters: {fn_parameters}
            Function Object:     {fn_object}

            Function
            --------\n{row}
            """
            if self._verbose:
                print(status_message)
            # And instantiate the object!
            self._module_dict[fn_id] = fn_object(**fn_parameters)
            # Now, call recursive init on *this* input to walk down
            #   the computation tree.
            self._recusive_init(id = ind)
            ########################################################
            # Holdover Code: Do not delete until the above is tested.
            # self.module_dict[fn_id] = atomic_constants[fn_i]
            # if fn_type == relu_i:
                
            # elif fn_type == linear_i:
            #     self.module_dict[fn_id] = nn.Linear(
            #             self.function_parameters["input_size"],
            #             self.function_parameters["output_size"],
            #     )
            # elif fn_type == dropout_i:
            #     self.module_dict[fn_id] = nn.Dropout(
            #         self.function_parameters["p"]
            #     )
        return

    def _recursive_forward(
        self,
        id: str,
        input: Tensor
    ) -> torch.Tensor:
        """Recursively call forward on internal layers.

        This uses the internal minimal experiment subspace and the
        module dictionary to call the forward. This recursively
        passes concatenated inputs to the next layer. This forward
        outputs the last layer's (prior to the sink) output.

        Parameters
        ----------
        id: str
            The ID of the layer in the experiment space.
        input: torch.Tensor
            The input data which is getting forward called on it.
            This is an m x len(self.inputs) tensor with each of the
            columns representing one of the inputs which goes into
            this composed function. This can be used to pull out
            required data from a dataset.
        """
        # Pull out the row for this ID
        data = self._net_subspace.query("id==@id")
        # Then get the inputs that feed into it.
        inputs = data.input.iloc[0]
        # And determine if it is a source node.
        is_source = data.type.iloc[0] == 'source'
        # If it's a source node we're simply going to return the
        #   column of the dataset that matches *this* input.
        if is_source:
            # This points at the correct input in the passed tensor.
            # This is assuming Torch input.
            input_data: torch.Tensor = map_to_output(
                data = input,
                input_mapping = self.inputs,
                ids = [id]
            ).float()
            return input_data
        # If it's *not* a source, then we recurse down each leg.
        other_inputs = [
            self._recursive_forward(inp, input) for inp in inputs
        ]
        # And concatenate the results together to make the input
        #   for this layer.
        input_data = torch.cat(
            other_inputs,
            # (
            #     input,
            #     *other_inputs
            # ),
            axis = 1
        )
        status_message = f"""Composed Function Status: Recursive Forward

        The ID of the forward function we are calling: {data.id.item()}
        The size of the input data: {input_data.shape}
        The Module we are calling: {self._module_dict[data.id.item()]}
        The input size of the module: {self._module_dict[data.id.item()].in_features}
        """

        # type_ = data.type.iloc[0]       # get the type of the input we're currently on
        # name = data.name.iloc[0]        # name of the input we're currently on

        # output = torch.zeros(1)         # cannot concat empty tensors, so this must be zeros(1)

        # output = output[1:]             # remove the added zeros(1) we created
        print(status_message)
        return self._module_dict[data.id.item()](input_data)


    def build_forward(
        self,
    ):
        """Build a PyTorch Graph.

        When building a PyTorch net we generally start at the input.
        This graph is easiest to *construct* by starting at the output
          and walking backward. When we've walked backwards using a
          recursion function we have a completed net.
        
        This function is using the ExperimentSpace to do this. This
        does not return anything, but it sets the self.model and
        self.inputs properties.

        """
        # This function is creating a ModuleDict to represent the
        # structure
        # Get the id of the sink:
        sink_id = self._net_subspace.query('type=="sink"').id.item()
        self._recusive_init(sink_id)
        # This needs to check the Module Dict
        if self._verbose:
            status_message = f"""Composed Build Completion Status:

            A composed function has been created.

            The PyTorch graph representation is:\n{self._module_dict}
            """
            if self._verbose: print(status_message)
    
    def forward(self, input: Tensor) -> Tensor:
        err_msg = """Forward Not Fully Implemented:

        This function should execute a PyTorch Graph.
        This should be calling recursive forward
        """
        # This is getting the 'tail end' of the graph.
        last_id = self._net_subspace.loc[
            self._net_subspace["type"] == "sink", "input"
        ].item()
        # This is then recursively walking back up the computation
        #   graph all the way to the inputs.
        last_out = self._recursive_forward(last_id, input)
        # last_id here because we haven't connected the output layer yet - nn.CrossEntropy or what have you
        # print(last_out)
        raise Exception(err_msg)

        # as a sanity check, last_out.requires_grad should be True...if it's not, then a comp graph wasn't built properly

        from torchviz import makedot

        makedot(last_out).render("comp_graph", format="png")    # visualize the output comp graph
        # --------------------------





        raise NotImplementedError(err_msg)

    def save(self):
        # Get the directory shorthand for readability
        d = self._function_dir
        # If it doesn't exist, make it.
        if not os.path.isdir(d):
            os.makedirs(d)
        # Save out the module dictionary
        # TODO: Does this work for composed functions of composed functions?
        torch.save(
            self._module_dict,
            os.path.join(d,'module_dict.pt')
        )
        # Save the inputs
        pd.Series(self.inputs).to_json(
            os.path.join(d,'inputs.json')
        )
        # and save the net subspace.
        self._net_subspace.to_json(
            os.path.join(d,'net_subspace.json')
        )

    def load(self):
        d = self._function_dir
        self._module_dict = torch.load(
            os.path.join(d,'module_dict.pt')
        )
        self.inputs = pd.read_json(
            'inputs.json',
            type='series',
            orient='records'
        ).to_dict()
        self._net_subspace = pd.read_json('net_subspace.json')

    def build_from_space(
        self,
        experiment_space: ExperimentSpace
    ) -> ExperimentSpace:
        """Create a minimal set of functions to build a graph.

        This, starting with the output node, will take an existing
        ExperimentSpace object and discard to only the required
        nodes which will be used to build the PyTorch computation
        graph. That will be stored as a class property.

        Parameters
        ----------
        experiment_space: ExperimentSpace
            This is a representation of Function objects.

        Returns
        -------
        minimal_experiment_space: ExperimentSpace
            This is a simplified representation of Function objects
            which will be used to construct a DaG. This downselects
            to *only* the information that is required to build the
            graph:
              * i
              * id
              * type
              * input
              * hyperparameters
              * object
        """
        persist_fields = [
            'id', 'type', 'input', 'hyperparameters', 'object'
        ]
        # TODO: Ensure this is not changing the input dataframe.
        # This line ensures that all the 'No input' nodes have None
        #   values, instead of NaN. This is used elsewhere, where None
        #   values will trigger a recursion stop.
        err_msg = """Composed Function Error:

        The Composed Function constructor was not given an
        ExperimentSpace object.
        """
        assert experiment_space is not None, err_msg
        experiment_space = experiment_space.fillna(
            np.nan
        ).replace([np.nan], [None])
        # Create new experiment space with only functions in the net.
        # This new space will have only intermediate and composite
        # functions in reverse order from output to input.
        net_df = pd.DataFrame().reindex(
            columns=experiment_space.columns
        )
        # This sets the first record of this space to be the *sink*
        #   node in the ExperimentSpace.
        net_df.loc[0] = experiment_space.query('type == "sink"').iloc[0]
        # Because the input is a *list* it must be recursively copied
        #   to prevent a 'pop' from affecting the primary dataset.
        cur_inputs = deepcopy(net_df.tail(1).input.item())
        # While the current inputs variable has values in it...
        while len(cur_inputs):
            # Snag the first one of those nodes and add it in to the
            #   output dataset.
            # cur_input = cur_inputs[0]
            cur_input = cur_inputs.pop(0)
            # But... only if it's not already in the output dataset.
            if cur_input in net_df.id.values:
                # If it already exists, skip this iteration.
                continue
            # if cur_input not in net_df.id.values:
            # This sets the *final row* of the output dataset to
            #   the output of querying the ExperimentSpace for the
            #   cur_input 'id' value. This should always return a
            #   single value.
            net_df.loc[len(net_df.index)] = experiment_space.query(
                "id == @cur_input"
            ).iloc[0]
            # This, then gets the *new* inputs for that downstream
            #   node.
            inps = net_df.tail(1).input.item()
            # And each of these inputs will get added to the list
            #   of inputs that we're wandering down.
            if inps is not None:
                for inp in inps:
                    # If it's not already in the output dataset, then
                    #   add it to the output dataset.
                    if (
                        inp not in net_df.id.values
                        # Deprecated.
                        # and experiment_space.query(
                        #     "id == @inp"
                        # ).type.values
                        # != "source"
                    ):
                        cur_inputs.append(inp)
        net_df = net_df[persist_fields]
        net_df.sort_values(
            by = ['id', 'type'],
            inplace = True
        )
        net_df = net_df.reset_index(drop=True)
        status_message = f"""Composed Function: build_from_space

        Input Experiment Space
        ----------------------\n{experiment_space}

        Output Minimal Experiment Space
        -------------------------------\n{net_df}
        """
        if self._verbose:
            print(status_message)
        # Now, assign the property.
        self._net_subspace = net_df


def map_to_output(
    data: Tensor,
    input_mapping: Dict[str, int],
    ids: Union[Iterable[str], str]
):
    """Subsets a tensor to provide the correct input data.

    Using a dict like {'input_0': 6, 'input_1': 3} put together
    an m x 2 dataset from a torch tensor.

    Parameters
    ----------
    data: torch.Tensor

    Returns
    -------
    subset: torch.Tensor
        A subset of the data, dependent on the IDs

    Examples
    --------
    >>> import torch
    >>> l = torch.tensor([
    ...     [1, 2, 3, 4, 5],
    ...     [3, 4, 5, 6, 7],
    ...     [5, 6, 7, 8, 9]
    ... ], dtype= torch.float)
    >>> d = {'input_0': 0, 'input_1': 3}
    >>> map_to_output(l, d, ['input_0'])
    tensor([[1.],
            [3.],
            [5.]], requires_grad=True)
    >>> map_to_output(l, d, ['input_0', 'input_1'])
    tensor([[1., 4.],
            [3., 6.],
            [5., 8.]], requires_grad=True)
    """
    # Ensure the list of strings is a list.
    if isinstance(ids, str):
        ids = [ids]
    # Then use that list to subset the columns, using the key
    #   mapping to pull the integer values for the named elements.
    subset = [v for k, v in input_mapping.items() if k in ids]
    layer = data[:, subset]
    return layer.clone().detach().requires_grad_(True)