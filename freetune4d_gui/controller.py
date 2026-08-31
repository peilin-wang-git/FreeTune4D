"""Framework-independent workflow state machine used by the GUI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .backend import PipelineConfig


class WorkflowState(str, Enum):
    INVALID = "invalid"
    READY = "ready"
    PREPROCESSING = "preprocessing"
    PREPROCESSED = "preprocessed"
    RECONSTRUCTING = "reconstructing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowController:
    state: WorkflowState = WorkflowState.INVALID
    config_signature: str | None = None
    preprocessing_valid: bool = False
    failure_stage: str | None = None

    def inputs_validated(self, config: PipelineConfig, errors: list[str]) -> None:
        signature = config.signature()
        if signature != self.config_signature:
            self.config_signature = signature
            self.preprocessing_valid = False
        if self.state not in {WorkflowState.PREPROCESSING, WorkflowState.RECONSTRUCTING}:
            self.state = WorkflowState.READY if not errors else WorkflowState.INVALID
            self.failure_stage = None

    def start_preprocessing(self) -> None:
        if self.state != WorkflowState.READY:
            raise RuntimeError("Preprocessing can only start from the ready state.")
        self.state = WorkflowState.PREPROCESSING
        self.preprocessing_valid = False

    def preprocessing_succeeded(self) -> None:
        if self.state != WorkflowState.PREPROCESSING:
            raise RuntimeError("Unexpected preprocessing completion.")
        self.state = WorkflowState.PREPROCESSED
        self.preprocessing_valid = True

    def start_reconstruction(self) -> None:
        if self.state != WorkflowState.PREPROCESSED or not self.preprocessing_valid:
            raise RuntimeError("Reconstruction requires successful preprocessing for the current case.")
        self.state = WorkflowState.RECONSTRUCTING

    def reconstruction_succeeded(self) -> None:
        if self.state != WorkflowState.RECONSTRUCTING:
            raise RuntimeError("Unexpected reconstruction completion.")
        self.state = WorkflowState.COMPLETED

    def operation_failed(self, stage: str) -> None:
        self.failure_stage = stage
        if stage == "preprocessing":
            self.preprocessing_valid = False
        self.state = WorkflowState.FAILED

    def recover_after_failure(self) -> None:
        if self.state != WorkflowState.FAILED:
            return
        if self.failure_stage == "reconstruction" and self.preprocessing_valid:
            self.state = WorkflowState.PREPROCESSED
        else:
            self.state = WorkflowState.READY

    @property
    def busy(self) -> bool:
        return self.state in {WorkflowState.PREPROCESSING, WorkflowState.RECONSTRUCTING}

    @property
    def can_preprocess(self) -> bool:
        return self.state == WorkflowState.READY

    @property
    def can_reconstruct(self) -> bool:
        return self.state == WorkflowState.PREPROCESSED and self.preprocessing_valid
