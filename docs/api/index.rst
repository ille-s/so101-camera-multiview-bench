API reference
=============

Hand-curated ``automodule`` listing for ``so101_mvbench``. Heavy dependencies
(Isaac Sim, Isaac Lab, torch, LeRobot) are mocked at documentation build time,
so signatures render without a simulator installed.

.. note::
   Every module of the package is listed below except these, and the omissions
   are deliberate rather than an oversight. ``tests/unit/test_docs_api_coverage.py``
   holds the same list and fails if a new module appears in neither place.

   * ``recording.teleop_recorder``, ``recording.multicam_replay``,
     ``evaluation.async_eval``, ``audit.determinism``, ``audit.cube_replay`` and
     ``tools.drift_configs`` are command modules. The first five boot Isaac Sim
     at import, and all six are documented as commands in
     :doc:`../getting-started/cli_overview`.
   * ``tasks.lift_cube_env_cfg`` is the Isaac-bound scene configuration, not an
     importable API.
   * ``assets.so101_white`` declares one articulation-config constant and has no
     module docstring, so ``automodule`` would render an empty section. Read the
     file itself, which carries its provenance in a comment header.

Recording (stage 1)
-------------------

.. automodule:: so101_mvbench.recording.recording_session

.. automodule:: so101_mvbench.recording.episode_controller

.. automodule:: so101_mvbench.recording.gamepad_action_generator

.. automodule:: so101_mvbench.recording.home_macro

.. automodule:: so101_mvbench.recording.recording_manifest

Assembly (stage 2)
------------------

.. automodule:: so101_mvbench.recording.trajectory_extractor

.. automodule:: so101_mvbench.assembly.bin_fusion

Training (stage 3)
------------------

.. automodule:: so101_mvbench.training.train_act_camera_perm

.. automodule:: so101_mvbench.training.checkpoint_selector

Evaluation (stages 4 and 5)
---------------------------

.. automodule:: so101_mvbench.evaluation.eval_tracker

.. automodule:: so101_mvbench.evaluation.eval_config

.. automodule:: so101_mvbench.evaluation.scene_loader

.. automodule:: so101_mvbench.evaluation.policy_wrapper

.. automodule:: so101_mvbench.evaluation.multienv_obs_utils

.. automodule:: so101_mvbench.evaluation.augmentation

.. automodule:: so101_mvbench.evaluation.results_writer

.. automodule:: so101_mvbench.evaluation.video_recorder

.. automodule:: so101_mvbench.evaluation.report

.. automodule:: so101_mvbench.evaluation.aggregate_report

.. automodule:: so101_mvbench.evaluation.plot_style

Audit tools
-----------

.. automodule:: so101_mvbench.audit.cube_plot

Dataset tools
-------------

.. automodule:: so101_mvbench.tools.list_envs

.. automodule:: so101_mvbench.tools.validate_bin

.. automodule:: so101_mvbench.tools.delete_episodes

.. automodule:: so101_mvbench.utils.dataset_manager

.. automodule:: so101_mvbench.utils.episodes_spec

Scene and geometry
------------------

.. automodule:: so101_mvbench.utils.so101_transforms

.. automodule:: so101_mvbench.utils.scene_config

.. automodule:: so101_mvbench.utils.scene_state

.. automodule:: so101_mvbench.utils.scene_builder

.. automodule:: so101_mvbench.utils.bin_spawner

.. automodule:: so101_mvbench.utils.camera_grid

.. automodule:: so101_mvbench.mdp.resets

Core
----

.. automodule:: so101_mvbench.settings

.. automodule:: so101_mvbench.logging
