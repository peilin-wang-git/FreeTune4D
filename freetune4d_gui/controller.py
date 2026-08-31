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
        signature_changed = signature != self.config_signature
        if signature_changed:
            self.config_signature = signature
            self.preprocessing_valid = False
        if self.state not in {WorkflowState.PREPROCESSING, WorkflowState.RECONSTRUCTING}:
            if errors:
                self.state = WorkflowState.INVALID
                self.preprocessing_valid = False
                self.failure_stage = None
            elif signature_changed or self.state == WorkflowState.INVALID:
                self.state = WorkflowState.READY
                self.failure_stage = None

    def start_preprocessing(self) -> None:
        retrying_failed_preprocessing = self.state == WorkflowState.FAILED and self.failure_stage == "preprocessing"
        if self.state != WorkflowState.READY and not retrying_failed_preprocessing:
            raise RuntimeError("Preprocessing can only start from the ready state.")
        self.state = WorkflowState.PREPROCESSING
        self.preprocessing_valid = False
        self.failure_stage = None

    def preprocessing_succeeded(self) -> None:
        if self.state != WorkflowState.PREPROCESSING:
            raise RuntimeError("Unexpected preprocessing completion.")
        self.state = WorkflowState.PREPROCESSED
        self.preprocessing_valid = True

    def start_reconstruction(self) -> None:
        retrying_failed_reconstruction = self.state == WorkflowState.FAILED and self.failure_stage == "reconstruction"
        if (self.state != WorkflowState.PREPROCESSED and not retrying_failed_reconstruction) or not self.preprocessing_valid:
            raise RuntimeError("Reconstruction requires successful preprocessing for the current case.")
        self.state = WorkflowState.RECONSTRUCTING
        self.failure_stage = None

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
        return self.state == WorkflowState.READY or (self.state == WorkflowState.FAILED and self.failure_stage == "preprocessing")

    @property
    def can_reconstruct(self) -> bool:
        valid_state = self.state == WorkflowState.PREPROCESSED or (self.state == WorkflowState.FAILED and self.failure_stage == "reconstruction")
        return valid_state and self.preprocessing_valid
