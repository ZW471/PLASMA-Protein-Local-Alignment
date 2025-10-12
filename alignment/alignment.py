from typing import Set, Literal

import torch
from torch import nn

from .interaction_nonlinearity import Hinge, LRL, DotProduct
from .interaction_structures import Sinkhorn

InteractionNonlinearity: Set[str] = {"hinge", "lrl"}
InteractionStructure: Set[str] = {"sinkhorn"}

class Alignment(nn.Module):
    def __init__(self, eta: InteractionNonlinearity, omega: InteractionStructure, eta_kwargs: dict = None, omega_kwargs: dict = None):
        """
        Initializes the object with specified interaction nonlinearity and interaction structure.
        It sets the attributes for `eta` and `omega` using the provided instances and their
        associated configuration dictionaries (`eta_kwargs` and `omega_kwargs`).

        :param eta: An instance representing the interaction nonlinearity.
        :param omega: An instance representing the interaction structure.
        :param eta_kwargs: Optional dictionary of configuration arguments for the `eta` attribute.
        :param omega_kwargs: Optional dictionary of configuration arguments for the `omega` attribute.
        """
        super().__init__()
        self.omega = None
        self.eta = None
        if eta_kwargs is None:
            eta_kwargs = {}
        if omega_kwargs is None:
            omega_kwargs = {}
        self.set_eta(eta, **eta_kwargs)
        self.set_omega(omega, **omega_kwargs)

    def set_eta(self, eta: InteractionNonlinearity, **kwargs) -> None:

        """
            Sets the interaction non-linearity for the instance. The method updates the
            non-linearity function based on the specified `eta` value. The supported
            non-linearity is "hinge", which sets the corresponding non-linearity
            function. If an unsupported value is provided for `eta`, a ValueError
            is raised.

            :param eta: Represents the interaction non-linearity. Expected value is
               "hinge".
            :type eta: InteractionNonlinearity
            :param kwargs: Additional keyword arguments. Currently unused but kept
               for extensibility.
            :return: None
            :rtype: None
            :raises ValueError: If an unsupported interaction non-linearity
               is provided in the `eta` parameter.
            """
        if eta == "hinge":
            self.eta = Hinge(**kwargs)
        elif eta == "lrl":
            self.eta = LRL(**kwargs)
        elif eta == "dot":
            self.eta = DotProduct(**kwargs)
        else:
            raise ValueError(f"Unknown interaction non-linearity: {eta}")

    def set_omega(self, omega: InteractionStructure, **kwargs) -> None:
        """
        Sets the interaction structure for the current instance.

        This method assigns an interaction structure to the ``omega`` attribute
        of the instance. If the interaction structure provided is "sinkhorn",
        it creates and sets a ``Sinkhorn`` instance with the provided keyword
        arguments. For any other interaction structure, a ``ValueError`` is
        raised to indicate an unknown or unsupported interaction structure.

        :param omega: The interaction structure to be assigned. If set to
            "sinkhorn", a ``Sinkhorn`` instance is created with the provided
            keyword arguments. Any other value will result in an exception.
        :type omega: InteractionStructure
        :param kwargs: Additional keyword arguments passed through to
            the ``Sinkhorn`` instance if "sinkhorn" is the specified
            interaction structure.
        :return: None
        :rtype: None
        :raises ValueError: If an unknown interaction structure is specified
            that is not "sinkhorn".
        """
        if omega == "sinkhorn":
            self.omega = Sinkhorn(**kwargs)
        elif omega == "none":
            self.omega = lambda x: x
        else:
            raise ValueError(f"Unknown interaction structure: {omega}")

    def forward(self, H_q: torch.Tensor, H_c: torch.Tensor, batch_q: torch.Tensor, batch_c: torch.Tensor) -> torch.Tensor:

        """
            Computes the forward pass for the given inputs by applying transformations using `eta`
            and `omega` functions. The method processes the input matrices and returns the
            transformed output.

            :param batch_c:
            :param batch_q:
            :param H_q: Query matrix, used as input to the transformation functions.
                It is expected to have the required dimensions compatible with `eta`.
            :param H_c: Context matrix, also used as input to the transformation functions.
                This is combined with `H_q` as part of the operations performed within `eta`.
            :return: The output resulting from applying the `omega` transformation to the
                processed values of `H_q` and `H_c` from `eta`.
            """
        return self.omega(self.eta(H_q=H_q, H_c=H_c, batch_q=batch_q, batch_c=batch_c))

    @staticmethod
    def reduce(alignment_matrix):
        return alignment_matrix.max(dim=1), alignment_matrix.max(dim=0)
