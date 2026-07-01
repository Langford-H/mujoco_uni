from __future__ import annotations

import subprocess
import sys
from typing import Any

import mujoco
import numpy as np
import pytest

from mujoco_uni.batch_env import BatchEnvPool

mj: Any = mujoco

PENDULUM_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" mass="1"/>
      <site name="tip" pos="0 0 -0.5"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" gear="2"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <jointvel joint="hinge"/>
    <framepos objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""

RAY_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="ball" pos="0.25 0 0.25">
      <geom name="ball_geom" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

HFIELD_XML = """
<mujoco>
  <asset>
    <hfield name="flat" nrow="2" ncol="2" size="1 1 0.5 0.1"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="flat"/>
    <body name="frame" pos="0 0 1">
      <geom type="sphere" size="0.01" mass="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model(xml: str = PENDULUM_XML) -> mujoco.MjModel:
  return mj.MjModel.from_xml_string(xml)


def _state_from_qpos_qvel(
    model: mujoco.MjModel, qpos: np.ndarray | None = None, qvel: np.ndarray | None = None
) -> np.ndarray:
  data = mj.MjData(model)
  if qpos is not None:
    data.qpos[:] = qpos
  if qvel is not None:
    data.qvel[:] = qvel
  mj.mj_forward(model, data)
  state = np.zeros(mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS))
  mj.mj_getState(model, data, state, int(mj.mjtState.mjSTATE_FULLPHYSICS))
  return state


def _batched_states(model: mujoco.MjModel, nbatch: int) -> np.ndarray:
  rows = []
  for i in range(nbatch):
    qpos = model.qpos0.copy()
    qvel = np.zeros(model.nv)
    if model.nq:
      qpos[0] += 0.05 * i
    if model.nv:
      qvel[0] += 0.02 * i
    rows.append(_state_from_qpos_qvel(model, qpos, qvel))
  return np.asarray(rows, dtype=np.float64)


def _serial_step(
    model: mujoco.MjModel,
    states: np.ndarray,
    *,
    nstep: int,
    control: np.ndarray | None = None,
    warmstart: np.ndarray | None = None,
    post_step_forward_sensor: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  out_state = np.zeros((states.shape[0], mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)))
  out_sensor = np.zeros((states.shape[0], model.nsensordata))
  control_spec = int(mj.mjtState.mjSTATE_CTRL)
  for i, state in enumerate(states):
    data = mj.MjData(model)
    mj.mj_setState(model, data, state, int(mj.mjtState.mjSTATE_FULLPHYSICS))
    if warmstart is not None:
      data.qacc_warmstart[:] = warmstart[i]
    for t in range(nstep):
      if control is not None:
        ctrl = control[i] if control.ndim == 2 else control[i, t]
        mj.mj_setState(model, data, ctrl, control_spec)
      mj.mj_step(model, data)
    mj.mj_getState(model, data, out_state[i], int(mj.mjtState.mjSTATE_FULLPHYSICS))
    if post_step_forward_sensor:
      mj.mj_setState(model, data, out_state[i], int(mj.mjtState.mjSTATE_FULLPHYSICS))
      data.ctrl[:] = 0
      data.qacc_warmstart[:] = 0
      mj.mj_forward(model, data)
    out_sensor[i] = data.sensordata
  return out_state, out_sensor


def _serial_forward(model: mujoco.MjModel, states: np.ndarray) -> np.ndarray:
  out = np.zeros((states.shape[0], model.nsensordata))
  for i, state in enumerate(states):
    data = mj.MjData(model)
    mj.mj_setState(model, data, state, int(mj.mjtState.mjSTATE_FULLPHYSICS))
    mj.mj_forward(model, data)
    out[i] = data.sensordata
  return out


def test_unsupported_mujoco_version_fails_before_batch_runtime_import() -> None:
  code = """
import mujoco
mujoco.__version__ = "2.3.7"
try:
    from mujoco_uni.batch_env import BatchEnvPool
except ImportError as exc:
    print(exc)
    raise SystemExit(0)
raise SystemExit("expected ImportError")
"""
  result = subprocess.run(
      [sys.executable, "-c", code],
      check=False,
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stdout + result.stderr
  assert "supports official mujoco>=3.8,<3.11" in result.stdout


def test_mujoco_build_runtime_mismatch_fails_fast() -> None:
  from mujoco_uni.compiled import MUJOCO_BUILD_VERSION

  assert MUJOCO_BUILD_VERSION is not None
  fake_runtime = "3.10.0"
  if str(MUJOCO_BUILD_VERSION).startswith("3.10."):
    fake_runtime = "3.8.0"

  code = f"""
import mujoco
mujoco.__version__ = {fake_runtime!r}
try:
    from mujoco_uni.batch_env import BatchEnvPool
except ImportError as exc:
    print(exc)
    raise SystemExit(0)
raise SystemExit("expected ImportError")
"""
  result = subprocess.run(
      [sys.executable, "-c", code],
      check=False,
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stdout + result.stderr
  assert "native batch extension was built against mujoco" in result.stdout
  assert "Rebuild mujoco_uni inside the selected MuJoCo environment" in result.stdout


def test_constructor_accepts_single_model_and_model_sequences() -> None:
  model = _model()
  with BatchEnvPool(model, nbatch=3, nthread=0) as pool:
    assert pool.nbatch == 3
    assert pool.nthread == 0
    assert len(pool.get_all_models()) == 3
    assert len(pool.get_model([0, 2])) == 2

  with BatchEnvPool([model], nbatch=2, nthread=1) as pool:
    assert pool.nbatch == 2

  with BatchEnvPool([model, model], nbatch=2, nthread=1) as pool:
    assert pool.nbatch == 2

  with pytest.raises(ValueError):
    BatchEnvPool(model, nbatch=0, nthread=0)
  with pytest.raises(ValueError):
    BatchEnvPool(model, nbatch=1, nthread=-1)
  with pytest.raises(ValueError):
    BatchEnvPool([model, model], nbatch=3, nthread=0)


@pytest.mark.parametrize("nthread,chunk_size", [(0, None), (1, None), (2, 1), (2, 5)])
def test_step_with_controls_and_sensors_matches_serial_reference(
    nthread: int, chunk_size: int | None
) -> None:
  model = _model()
  nbatch = 4
  nstep = 5
  states = _batched_states(model, nbatch)
  warmstart = np.linspace(0.0, 0.03, nbatch * model.nv).reshape(nbatch, model.nv)
  ncontrol = mj.mj_stateSize(model, mj.mjtState.mjSTATE_CTRL)
  control = np.zeros((nbatch, nstep, ncontrol))
  for i in range(nbatch):
    control[i, :, 0] = np.linspace(0.1 * i, 0.1 * i + 0.2, nstep)

  expected_state, expected_sensor = _serial_step(
      model, states, nstep=nstep, control=control, warmstart=warmstart
  )
  with BatchEnvPool(model, nbatch=nbatch, nthread=nthread) as pool:
    got_state, got_sensor = pool.step(
        states,
        nstep=nstep,
        control=control,
        initial_warmstart=warmstart,
        chunk_size=chunk_size,
        return_sensor=True,
    )

  np.testing.assert_allclose(got_state, expected_state, atol=1e-12, rtol=0)
  np.testing.assert_allclose(got_sensor, expected_sensor, atol=1e-12, rtol=0)


def test_step_accepts_constant_control_and_post_forward_sensor_refresh() -> None:
  model = _model()
  nbatch = 3
  nstep = 4
  states = _batched_states(model, nbatch)
  ncontrol = mj.mj_stateSize(model, mj.mjtState.mjSTATE_CTRL)
  control = np.full((nbatch, ncontrol), 0.15)

  expected_state, expected_sensor = _serial_step(
      model,
      states,
      nstep=nstep,
      control=control,
      post_step_forward_sensor=True,
  )
  with BatchEnvPool(model, nbatch=nbatch, nthread=2) as pool:
    got_state, got_sensor = pool.step(
        states,
        nstep=nstep,
        control=control,
        return_sensor=True,
        post_step_forward_sensor=True,
        chunk_size=2,
    )

  np.testing.assert_allclose(got_state, expected_state, atol=1e-12, rtol=0)
  np.testing.assert_allclose(got_sensor, expected_sensor, atol=1e-12, rtol=0)


def test_forward_matches_serial_reference() -> None:
  model = _model()
  states = _batched_states(model, 4)
  expected = _serial_forward(model, states)

  with BatchEnvPool(model, nbatch=4, nthread=2) as pool:
    got = pool.forward(states, chunk_size=2)

  np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


def test_reset_randomization_and_field_indexing_are_isolated() -> None:
  model = _model()
  states = _batched_states(model, 3)
  reset_ids = [2, 0]
  reset_states = states[reset_ids].copy()

  with BatchEnvPool(model, nbatch=3, nthread=2) as pool:
    original_env1_gravity = pool.get_field(1, "gravity").copy()
    gravity = np.asarray([[0.0, 0.0, -5.0], [0.0, 0.0, -3.0]])
    state_out, sensor_out = pool.reset(
        reset_ids,
        reset_states,
        randomization={"gravity": gravity},
        chunk_size=1,
    )

    assert state_out.shape == (2, pool.nstate)
    assert sensor_out.shape == (2, pool.nsensordata)
    np.testing.assert_allclose(pool.get_field(2, "gravity"), gravity[0])
    np.testing.assert_allclose(pool.get_field(0, "gravity"), gravity[1])
    np.testing.assert_allclose(pool.get_field(1, "gravity"), original_env1_gravity)

    mass0 = pool.get_field_indexed(0, "body_mass", 0)
    pool.set_field_indexed(0, "body_mass", 0, mass0 + 0.125)
    assert pool.get_field_indexed(0, "body_mass", 0) == pytest.approx(mass0 + 0.125)

    ipos = pool.get_field_indexed(0, "body_ipos", 0)
    updated = ipos + np.asarray([0.01, -0.02, 0.03])
    pool.set_field_indexed(0, "body_ipos", 0, updated)
    np.testing.assert_allclose(pool.get_field_indexed(0, "body_ipos", 0), updated)


def test_compute_site_jacobians_matches_serial_reference() -> None:
  model = _model()
  states = _batched_states(model, 3)
  site_id = mj.mj_name2id(model, int(mj.mjtObj.mjOBJ_SITE), "tip")
  assert site_id >= 0

  expected_jp = np.zeros((3, 3, model.nv))
  expected_jr = np.zeros((3, 3, model.nv))
  for i, state in enumerate(states):
    data = mj.MjData(model)
    mj.mj_setState(model, data, state, int(mj.mjtState.mjSTATE_FULLPHYSICS))
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    mj.mj_jacSite(model, data, expected_jp[i], expected_jr[i], site_id)

  with BatchEnvPool(model, nbatch=3, nthread=2) as pool:
    got_jp, got_jr = pool.compute_site_jacobians(
        states, site_id, jacp=True, jacr=True, chunk_size=2
    )

  np.testing.assert_allclose(got_jp, expected_jp, atol=1e-12, rtol=0)
  np.testing.assert_allclose(got_jr, expected_jr, atol=1e-12, rtol=0)


def test_multi_ray_matches_serial_reference() -> None:
  model = _model(RAY_XML)
  states = _batched_states(model, 2)
  pnt = np.asarray([0.0, 0.0, 1.0])
  vec = np.asarray([[0.0, 0.0, -1.0], [0.25, 0.0, -1.0]], dtype=np.float64)
  nray = vec.shape[0]

  expected_geomid = np.zeros((2, nray), dtype=np.int32)
  expected_dist = np.zeros((2, nray), dtype=np.float64)
  expected_normal = np.zeros((2, nray, 3), dtype=np.float64)
  vec_flat = vec.reshape(-1)
  for i, state in enumerate(states):
    data = mj.MjData(model)
    mj.mj_setState(model, data, state, int(mj.mjtState.mjSTATE_FULLPHYSICS))
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    normal_flat = np.zeros(nray * 3, dtype=np.float64)
    mj.mj_multiRay(
        model,
        data,
        pnt,
        vec_flat,
        None,
        1,
        -1,
        expected_geomid[i],
        expected_dist[i],
        normal_flat,
        nray,
        float(mj.mjMAXVAL),
    )
    expected_normal[i] = normal_flat.reshape(nray, 3)

  with BatchEnvPool(model, nbatch=2, nthread=2) as pool:
    got_dist, got_geomid, got_normal = pool.multi_ray(
        states, pnt, vec, return_normal=True, chunk_size=1
    )

  np.testing.assert_array_equal(got_geomid, expected_geomid)
  np.testing.assert_allclose(got_dist, expected_dist, atol=1e-12, rtol=0)
  np.testing.assert_allclose(got_normal, expected_normal, atol=1e-12, rtol=0)


def test_hfield_height_sampling_flat_terrain() -> None:
  model = _model(HFIELD_XML)
  hfield_geom_id = mj.mj_name2id(model, int(mj.mjtObj.mjOBJ_GEOM), "terrain")
  frame_body_id = mj.mj_name2id(model, int(mj.mjtObj.mjOBJ_BODY), "frame")
  assert hfield_geom_id >= 0
  assert frame_body_id >= 0
  states = _batched_states(model, 2)
  offsets = np.asarray([[0.0, 0.0], [0.5, -0.5], [2.0, 2.0]], dtype=np.float64)

  with BatchEnvPool(model, nbatch=2, nthread=2) as pool:
    height = pool.sample_hfield_height(
        states, hfield_geom_id, offsets, frame_body_id, output="height", chunk_size=1
    )
    clearance = pool.sample_hfield_height(
        states, hfield_geom_id, offsets, frame_body_id, output="clearance", chunk_size=1
    )

  np.testing.assert_allclose(height, 0.0, atol=1e-12, rtol=0)
  np.testing.assert_allclose(clearance, 1.0, atol=1e-12, rtol=0)


def test_public_methods_validate_chunk_size_and_closed_pool() -> None:
  model = _model()
  states = _batched_states(model, 2)
  site_id = mj.mj_name2id(model, int(mj.mjtObj.mjOBJ_SITE), "tip")
  with BatchEnvPool(model, nbatch=2, nthread=2) as pool:
    with pytest.raises(ValueError, match="chunk_size"):
      pool.step(states, nstep=1, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
      pool.forward(states, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
      pool.compute_site_jacobians(states, site_id, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
      pool.reset([0], states[:1], chunk_size=0)

  with pytest.raises(RuntimeError, match="after pool close"):
    pool.forward(states)


def test_repeated_create_step_and_close_loop_stays_finite() -> None:
  model = _model()
  nbatch = 8
  ncontrol = mj.mj_stateSize(model, mj.mjtState.mjSTATE_CTRL)

  for _ in range(5):
    states = _batched_states(model, nbatch)
    control = np.zeros((nbatch, ncontrol))
    pool = BatchEnvPool(model, nbatch=nbatch, nthread=2)
    try:
      for _ in range(10):
        states = pool.step(states, nstep=2, control=control, chunk_size=3)
        assert states.shape == (nbatch, pool.nstate)
        assert np.isfinite(states).all()
    finally:
      pool.close()

    pool.close()
