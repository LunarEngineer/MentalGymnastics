import json
import os
import pandas as pd
import torch

from mentalgym.types import ExperimentSpace, FunctionBank
from typing import Any, Dict, Iterable, Optional, Union

class ComposedFunction():
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
        function_bank: Optional[FunctionBank] = None
    ):
        # 1) Is this building the net? We will check to see if the
        #   Function's directory exists.
        function_dir = os.path.join(
            experiment_space._function_bank_directory,
            id
        )
        folder_exists = os.path.isdir(function_dir)
        if not folder_exists:
            self.build_forward(experiment_space, function_bank)
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

