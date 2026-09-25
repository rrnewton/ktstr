//! A ktstr scenario run on scx_simulator IN-PROCESS: exported as the typed
//! record, lowered by scxsim-workload-ir and simulated inside this test binary.
//! No record is written to disk and no simulator binary is spawned. This is the
//! path that replaces the exported-file bridge (`KTSTR_SCENARIO_EXPORT_DIR`).
//!
//! Built only with the `scxsim` feature. Its build.rs compiles the `simple`
//! scheduler `.so` into `KTSTR_SCXSIM_SO_DIR` and gives `[[test]]` binaries the
//! host link arguments scx_simulator checks for before it loads one.
//!
//! No `KtstrTestEntry` and no declared scheduler, so ktstr's nextest dispatch
//! interception never fires and the standard harness runs the `#[test]`s.

use std::time::Duration;

use ktstr::scenario::ScenarioDef;
use ktstr::scenario::export::export_scenario;
use ktstr::scenario::ops::CgroupDef;
use ktstr::test_support::Topology;
use scx_simulator::prelude::*;

const WORKERS_PER_CGROUP: u32 = 2;

fn topology() -> Topology {
    Topology {
        llcs: 1,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
        llc_cores: None,
    }
}

/// The `sched_basic_proportional` shape: two cgroups of default workers,
/// exported, lowered and run to completion on `simple`.
#[test]
fn an_exported_scenario_runs_in_process() {
    let def = ScenarioDef::with_defs(vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")]);
    let out = export_scenario(
        "in_process",
        &def,
        &topology(),
        Duration::from_secs(1),
        WORKERS_PER_CGROUP,
    );
    assert!(out.is_complete(), "unexpected gaps: {:#?}", out.gaps);

    let ir = scxsim_workload_ir::lower(&out.record).expect("the lowering accepts the record");
    let scenario = scxsim_workload_ir::to_scenario(&ir).expect("the IR ingests");
    let pids: Vec<Pid> = scenario.tasks.iter().map(|t| t.pid).collect();
    assert_eq!(
        pids.len(),
        2 * WORKERS_PER_CGROUP as usize,
        "the simulated workload is not the exported one: {:#?}",
        scenario.tasks,
    );

    let so = concat!(env!("KTSTR_SCXSIM_SO_DIR"), "/libscx_simple.so");
    // The compiled scheduler has process-global state: one simulation at a time.
    let _guard = scx_simulator::SIM_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let sched = DynamicScheduler::try_load(so, "simple", scenario.nr_cpus)
        .unwrap_or_else(|e| panic!("load {so}: {e}"));
    let trace = Simulator::new(sched).run(scenario);

    assert_eq!(trace.exit_kind(), &ExitKind::Normal);
    for pid in pids {
        assert!(
            trace.schedule_count(pid) > 0,
            "task {pid:?} was never scheduled"
        );
    }
}
