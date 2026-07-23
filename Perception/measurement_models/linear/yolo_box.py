import numpy as np

from .linear_base import LinearMeasurementModel



class BoundingBoxMeasurementModel(LinearMeasurementModel):
    """
    Linear measurement model for YOLO bounding boxes.

    Raw detector representation:

        m = [x1, y1, x2, y2]

    Filter measurement representation:

        z = [cx, cy, width, height]

    The hidden state must begin with:

        x = [cx, cy, width, height, ...]

    Examples
    --------
    Constant-velocity image-space state:

        [cx, cy, width, height,
         vx, vy, width_rate, height_rate]

    Constant-acceleration image-space state:

        [cx, cy, width, height,
         vx, vy, width_rate, height_rate,
         ax, ay, width_acceleration, height_acceleration]
    """

    name = "YOLO bounding-box measurement"
    measurement_dim = 4

    def __init__(
        self,
        state_dim: int,
        corner_covariance: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        state_dim:
            Dimension of the hidden filter state.

            The first four state components must be:

                [cx, cy, width, height]

        corner_covariance:
            Covariance matrix Sigma_m of the raw YOLO box errors:

                [x1_error, y1_error, x2_error, y2_error]

            Expected shape:

                (4, 4)
        """
        if not isinstance(state_dim, int):
            raise TypeError("state_dim must be an integer.")

        if state_dim < self.measurement_dim:
            raise ValueError(
                "state_dim must be at least 4 because the state must "
                "contain [cx, cy, width, height]."
            )

        sigma_m = np.asarray(
            corner_covariance,
            dtype=float,
        ).copy()

        if sigma_m.shape != (4, 4):
            raise ValueError(
                "corner_covariance must have shape (4, 4), "
                f"but received {sigma_m.shape}."
            )

        if not np.all(np.isfinite(sigma_m)):
            raise ValueError(
                "corner_covariance contains NaN or infinite values."
            )

        if not np.allclose(
            sigma_m,
            sigma_m.T,
            atol=1e-10,
        ):
            raise ValueError(
                "corner_covariance must be symmetric."
            )

        # Remove only insignificant floating-point asymmetry
        # after confirming that the input is intended to be symmetric.
        sigma_m = 0.5 * (sigma_m + sigma_m.T)

        eigenvalues = np.linalg.eigvalsh(sigma_m)

        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                "corner_covariance must be positive semidefinite."
            )

        self.state_dim = state_dim
        self.corner_covariance = sigma_m

        # Converts:
        #
        #     [x1, y1, x2, y2]
        #
        # into:
        #
        #     [cx, cy, width, height]
        self._corner_transform = np.array(
            [
                [0.5,  0.0,  0.5,  0.0],
                [0.0,  0.5,  0.0,  0.5],
                [-1.0, 0.0,  1.0,  0.0],
                [0.0, -1.0,  0.0,  1.0],
            ],
            dtype=float,
        )

        # H selects [cx, cy, width, height] from the state.
        self._H = np.zeros(
            (self.measurement_dim, self.state_dim),
            dtype=float,
        )

        self._H[:, :self.measurement_dim] = np.eye(
            self.measurement_dim,
            dtype=float,
        )

        # Propagate raw corner-coordinate uncertainty into
        # center-size measurement space:
        #
        #     R = A Sigma_m A^T
        self._R = (
            self._corner_transform
            @ self.corner_covariance
            @ self._corner_transform.T
        )

        # Remove tiny floating-point asymmetry.
        self._R = 0.5 * (self._R + self._R.T)

        if not np.all(np.isfinite(self._R)):
            raise ValueError(
                "Computed measurement covariance R contains NaN or infinity."
            )

        if np.any(np.linalg.eigvalsh(self._R) < -1e-10):
            raise ValueError(
                "Computed measurement covariance R must be positive semidefinite."
            )

    def measurement_matrix(self) -> np.ndarray:
        """
        Return the linear measurement matrix H.

        The model observes the first four state variables:

            [cx, cy, width, height]
        """
        return self._H.copy()

    def measurement_noise(self) -> np.ndarray:
        """
        Return the measurement-noise covariance R.
        """
        return self._R.copy()

    def raw_to_measurement(
        self,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a YOLO box from corner form into center-size form.

        Parameters
        ----------
        bbox:
            Bounding box:

                [x1, y1, x2, y2]

        Returns
        -------
        np.ndarray
            Measurement:

                [cx, cy, width, height]
        """
        raw_box = np.asarray(
            bbox,
            dtype=float,
        ).reshape(-1)

        if raw_box.shape != (4,):
            raise ValueError(
                "bbox must contain exactly four values "
                "[x1, y1, x2, y2]."
            )

        if not np.all(np.isfinite(raw_box)):
            raise ValueError(
                "bbox contains NaN or infinite values."
            )

        x1, y1, x2, y2 = raw_box

        if x2 <= x1:
            raise ValueError("bbox requires x2 > x1.")

        if y2 <= y1:
            raise ValueError("bbox requires y2 > y1.")

        measurement = self._corner_transform @ raw_box
        return measurement.copy()