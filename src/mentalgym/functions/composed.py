import json
import os
import torch

class ComposedFunction():
    """Composed of multiple atomic functions.

    This class is used to build a representation for a composed
    function. It can be instantiated in one of two ways:

    1. When called for the first time this will be provided a folder
       and an experiment space; the space will be used to build the
       PyTorch graph. The graph will be persisted. The function bank
       passed will be used to query for information.
    2. When called subsequently this will only be provided a folder
       path and a function bank; it will build itself from that
       location.

    Parameters
    ----------
    fn_path: str
        This is the location
    """
    def __init__(
        self,
        fn_path: str,
        experiment_space: Optional[ExperimentSpace] = None,
        function_bank: Optional[FunctionBank] = None
    ):
        self.fn_path = fn_path
        self._function_bank = function_bank
        self._experiment_space = experiment_space
        # 1) Is this building the net?
        has_space = experiment_space is not None
        has_bank = function_bank is not None
        if has_bank:
            if has_space:
                self.model = self.build_from_space()
            else:
                self.model = self.load()
        else:
            try:
                self.model = self.load()
            except:
                err_msg = """Composed Function Error:

                The Composed Function constructor was not given a
                FunctionBank object; when it attempted to pull a
                model from the given location it was unable to do so.

                Function Bank
                -------------\n{function_bank}

                Experiment Space
                ----------------\n{experiment_space}

                Function Directory
                ------------------\n{fn_path}
                """
                raise Exception(err_msg)

    def build_from_space():
        """Builds a net from an experiment space representation.

        Returns
        -------
        model: torch.module
            This is a PyTorch layer with forward, backward, etc...

        """

        # When building a PyTorch net we generally start at the input.
        # This graph is easiest to *construct* by starting at the output
        #   and walking backward. When we've walked backwards using a
        #   recursion function we have a nested dictionary of input IDs.
        # We can build the net from that nested dictionary with another
        #   recursive function.
    def save(self, repr, model):
        with open(os.path.join(self.fn_path, "connectivity_graph.json"), 'w') as f:
            f.write(json.dumps(repr))
        torch.save(model.state_dict(), os.path.join(self.fn_path, "weights.pt"))

    def load(self):
        with open(os.path.join(self.fn_path, "connectivity_graph.json"), 'r') as f:
            self.repr = f.read()
        torch.load(os.path.join(self.fn_path, "weights.pt"))

