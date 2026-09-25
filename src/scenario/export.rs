//! Export a [`ScenarioDef`] as a backend-neutral workload record.
//!
//! [`ScenarioDef`] made a test's workload a value; this makes it a value the
//! simulator can take. The record is `scxsim-workload-ir`'s
//! [`SourceScenario`] itself, which lowers to a restricted IR and is ingested
//! by the simulator — so the same test definition can drive the VM backend and
//! the simulator without being written twice.
//!
//! # Why the record is the IR crate's type and not JSON
//!
//! This file used to assemble the record with `serde_json::json!`, matching
//! `SourceScenario`'s serde form by hand, because ktstr had no way to name the
//! type: it lived only in the sched-test repository, and a path dependency
//! between two checkouts would bake a host-specific path into a committed
//! manifest. JSON was the seam two repos could share without either owning the
//! other's build, and a drift surfaced only when the simulator side
//! deserialised a record and named the field.
//!
//! With `scxsim-workload-ir` an ordinary crates.io dependency, the record is
//! built as the type and the drift moves to compile time, in this file. Its
//! default features are serde and serde_json only — no simulator, no BPF build
//! — so taking the type costs ktstr nothing it did not already link. The JSON
//! written by `export_registered_scenarios` is still produced, now as the
//! type's own serde form, for tools that consume records out of process.
//!
//! # What is deliberately not exported
//!
//! Anything the record cannot represent faithfully is omitted with a recorded
//! reason rather than approximated here. Approximation is the *lowering's* job,
//! where it is classified and reported in the
//! [`FidelityReport`](scxsim_workload_ir::FidelityReport) the IR carries; an
//! approximation invented at export time would be invisible to that machinery
//! and would reach the simulator disguised as an exact reading of the test.

use scxsim_workload_ir::{
    DurationNs, SourceCgroupDef, SourceCpuset, SourceHold, SourceScenario, SourceStep,
    SourceTopology, SourceWorkSpec, SourceWorkType,
};
use serde::Serialize;

use crate::scenario::ScenarioDef;
use crate::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Setup, Step};
use crate::test_support::Topology;
use crate::workload::{WorkSpec, WorkType};

/// A reason some part of a scenario could not be exported faithfully.
///
/// Carried out of [`export_scenario`] rather than logged, for the same reason
/// the IR carries its fidelity report: a caller deciding whether a simulator
/// run answers its question needs to see what the record does not say.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportGap {
    /// Where in the scenario the gap is, e.g. `step[1].setup[0] "cg_0"`.
    pub where_: String,
    /// The construct that could not be exported.
    pub construct: String,
    /// Why, in terms a reader can act on.
    pub reason: String,
}

/// The exported record plus everything it could not carry.
#[derive(Debug, Clone)]
pub struct Export {
    /// The scenario in the simulator's source vocabulary.
    pub record: SourceScenario,
    /// Constructs omitted from `record`, each with a reason.
    pub gaps: Vec<ExportGap>,
}

impl Export {
    /// Whether the record carries the scenario in full.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.gaps.is_empty()
    }
}

/// ktstr's `CpusetSpec` -> `SourceCpuset`.
///
/// Returns why a variant cannot be carried, for the caller to record as a gap.
/// Resolving one here instead (say, by flattening a topology-relative cpuset
/// into an explicit CPU list) would bind the scenario to this host's topology,
/// which is exactly the symbolic-ness the DSL exists to keep.
fn cpuset(spec: &CpusetSpec) -> Result<SourceCpuset, &'static str> {
    // ktstr indexes with usize and the record with u32. No topology comes near
    // the difference, but an index that does not fit is reported rather than
    // truncated into a different, plausible cpuset.
    let n = |v: usize| {
        u32::try_from(v).map_err(|_| {
            "an index or count outside u32, which the record cannot carry; \
             truncating would silently name a different cpuset"
        })
    };
    Ok(match spec {
        CpusetSpec::Llc(i) => SourceCpuset::Llc(n(*i)?),
        CpusetSpec::Numa(i) => SourceCpuset::Numa(n(*i)?),
        CpusetSpec::Disjoint { index, of } => SourceCpuset::Disjoint {
            index: n(*index)?,
            of: n(*of)?,
        },
        CpusetSpec::Range {
            start_frac,
            end_frac,
        } => SourceCpuset::Range {
            start_frac: *start_frac,
            end_frac: *end_frac,
        },
        CpusetSpec::Overlap { index, of, frac } => SourceCpuset::Overlap {
            index: n(*index)?,
            of: n(*of)?,
            frac: *frac,
        },
        _ => {
            return Err("no SourceCpuset counterpart; resolving it here would \
                        bind the scenario to this host's topology");
        }
    })
}

/// ktstr's `WorkType` -> `SourceWorkType`.
///
/// Only the variants whose scheduler-visible meaning transfers verbatim are
/// mapped. Everything else is a gap: the lowering is the layer that decides how
/// a work type collapses onto run/sleep/yield and records the discarded
/// dimension, and it cannot do that for a construct this function has already
/// silently rewritten.
///
/// # Why this covers the FIELDLESS variants and no others
///
/// `SourceWorkType` mirrors this enum — same 45 names — so it is tempting to
/// map all 45 mechanically. Do not. The mirror is not exact: several variants
/// carry ktstr fields the IR has no home for, among them
/// `PriorityInversion::pi_mode`, `ProducerConsumerImbalance::queue_depth_target`
/// and `Custom::{run, cfg}`. Mapping those by name would drop a tuning knob
/// silently, which is the one thing this file exists not to do.
///
/// The nine fieldless variants have nothing to drop, so they transfer
/// verbatim and are safe. Two field-carrying variants are also mapped --
/// `FutexPingPong` and `CrossAffinityChurn` -- each checked field-by-field
/// against the IR first; both carry only `spin_iters: u64` on both sides. The
/// remaining 34 want a per-variant mapping that translates the fields it can
/// and records an [`ExportGap`] for the fields it cannot — worth doing, but it
/// is a per-variant judgement each time, not a loop.
fn work_type(wt: &WorkType) -> Option<SourceWorkType> {
    Some(match wt {
        WorkType::SpinWait => SourceWorkType::SpinWait,
        WorkType::YieldHeavy => SourceWorkType::YieldHeavy,
        WorkType::Mixed => SourceWorkType::Mixed,
        // FIELDLESS variants, verbatim and only verbatim: each names the same
        // fieldless `SourceWorkType`, so nothing is rewritten here. What the
        // simulator cannot model about them -- for `IoSyncWrite`, the block
        // device, queue depth and byte counts -- is discarded one layer down by
        // the lowering, which records the cause when it does.
        WorkType::IoSyncWrite => SourceWorkType::IoSyncWrite,
        WorkType::IoRandRead => SourceWorkType::IoRandRead,
        WorkType::IoConvoy => SourceWorkType::IoConvoy,
        WorkType::ForkExit => SourceWorkType::ForkExit,
        WorkType::NiceSweep => SourceWorkType::NiceSweep,
        WorkType::SmtSiblingSpin => SourceWorkType::SmtSiblingSpin,

        // FIELD-CARRYING variants, mapped one at a time with their fields
        // checked against the IR rather than by name. Both of these carry a
        // single `spin_iters: u64` on BOTH sides, so nothing is dropped and no
        // gap is recorded -- which is the bar a field-carrying mapping has to
        // meet before it belongs here. A variant whose ktstr fields have no IR
        // home must record an ExportGap instead; see the note above about
        // PriorityInversion::pi_mode and friends.
        WorkType::FutexPingPong { spin_iters } => SourceWorkType::FutexPingPong {
            spin_iters: *spin_iters,
        },
        WorkType::CrossAffinityChurn { spin_iters } => SourceWorkType::CrossAffinityChurn {
            spin_iters: *spin_iters,
        },

        _ => return None,
    })
}

fn work_spec(w: &WorkSpec, at: &str, gaps: &mut Vec<ExportGap>) -> Option<SourceWorkSpec> {
    let Some(wt) = work_type(&w.work_type) else {
        gaps.push(ExportGap {
            where_: at.to_string(),
            construct: format!("WorkType::{:?}", w.work_type),
            reason: "no SourceWorkType counterpart is mapped yet; the lowering, \
                     not this exporter, is where a work type is approximated and \
                     the discarded dimension recorded"
                .to_string(),
        });
        return None;
    };
    // `nice` used to be hardcoded null here while `WorkSpec` carried one and
    // `SourceWorkSpec` had a field waiting for it — a silent drop, in the file
    // whose whole premise is that it is allowed to be wrong but not quietly
    // wrong. It went unnoticed because every scenario ported so far leaves nice
    // at the default.
    //
    // The widths differ (ktstr i32, IR i8). Linux nice is -20..=19 so every
    // legal value fits, and an out-of-range one is recorded as a gap rather
    // than truncated into a different, plausible priority.
    let nice = match w.nice {
        None => None,
        Some(n) => match i8::try_from(n) {
            Ok(v) => Some(v),
            Err(_) => {
                gaps.push(ExportGap {
                    where_: at.to_string(),
                    construct: format!("WorkSpec::nice = {n}"),
                    reason: "outside i8, so outside the -20..=19 nice range the \
                             record can express; truncating would silently \
                             substitute a different priority"
                        .to_string(),
                });
                None
            }
        },
    };

    // `sched_policy` has no counterpart in `SourceWorkSpec` at all, so this is
    // a gap in the SCHEMA rather than in this function — but it was previously
    // not even mentioned, which made a policy-mixing scenario export as though
    // every worker were SCHED_NORMAL. `custom_sched_mixed` is exactly that
    // scenario: its point is a Normal/Batch/Idle/FIFO mix.
    if w.sched_policy != Default::default() {
        gaps.push(ExportGap {
            where_: at.to_string(),
            construct: format!("WorkSpec::sched_policy = {:?}", w.sched_policy),
            reason: "SourceWorkSpec has no scheduling-policy field, so the \
                     record cannot distinguish SCHED_NORMAL from BATCH, IDLE, \
                     FIFO or DEADLINE; a policy-mixing scenario would otherwise \
                     export as uniformly SCHED_NORMAL"
                .to_string(),
        });
    }

    Some(SourceWorkSpec {
        workers: w.num_workers.map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
        work_type: wt,
        nice,
    })
}

fn cgroup_def(def: &CgroupDef, at: &str, gaps: &mut Vec<ExportGap>) -> SourceCgroupDef {
    let cs = match &def.cpuset {
        None => None,
        Some(spec) => match cpuset(spec) {
            Ok(c) => Some(c),
            Err(reason) => {
                gaps.push(ExportGap {
                    where_: at.to_string(),
                    construct: format!("CpusetSpec::{spec:?}"),
                    reason: reason.to_string(),
                });
                None
            }
        },
    };

    // `works` empty means "one default WorkSpec", which is what the step runner
    // resolves it to (`merged_works`). Make that explicit in the record rather
    // than exporting an empty list the consumer would have to know to reinterpret.
    let works: Vec<SourceWorkSpec> = if def.works.is_empty() {
        work_spec(&WorkSpec::default(), at, gaps)
            .into_iter()
            .collect()
    } else {
        def.works
            .iter()
            .filter_map(|w| work_spec(w, at, gaps))
            .collect()
    };

    if def.payload.is_some() {
        gaps.push(ExportGap {
            where_: at.to_string(),
            construct: "CgroupDef::payload".to_string(),
            reason: "a payload runs an external binary; its behaviour is not \
                     derivable from the declaration"
                .to_string(),
        });
    }

    SourceCgroupDef {
        name: def.name.to_string(),
        cpuset: cs,
        works,
        cpu_quota: None,
        cpu_weight: None,
    }
}

/// Durations become [`DurationNs`], which saturates at `u64::MAX` ns rather
/// than wrapping; the record has always had that ceiling.
fn hold(h: &HoldSpec) -> SourceHold {
    match h {
        HoldSpec::Frac(f) => SourceHold::Frac(*f),
        HoldSpec::Fixed(d) => SourceHold::Fixed(DurationNs::from_std(*d)),
        HoldSpec::Loop { interval } => SourceHold::Loop {
            interval: DurationNs::from_std(*interval),
        },
    }
}

fn step(s: &Step, idx: usize, gaps: &mut Vec<ExportGap>) -> SourceStep {
    let setup: Vec<SourceCgroupDef> = match &s.setup {
        Setup::Defs(defs) => defs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let at = format!("step[{idx}].setup[{i}] {:?}", d.name.as_ref());
                cgroup_def(d, &at, gaps)
            })
            .collect(),
        Setup::Factory(_) => {
            gaps.push(ExportGap {
                where_: format!("step[{idx}].setup"),
                construct: "Setup::Factory".to_string(),
                reason: "the cgroup list is produced by an fn(&Ctx) and cannot be \
                         evaluated without a running guest"
                    .to_string(),
            });
            Vec::new()
        }
    };

    if !s.ops.is_empty() {
        gaps.push(ExportGap {
            where_: format!("step[{idx}].ops"),
            construct: format!("{} Op(s)", s.ops.len()),
            reason: "ops are not exported yet; the record would describe a \
                     scenario missing its mutations"
                .to_string(),
        });
    }

    SourceStep {
        setup,
        ops: Vec::new(),
        hold: hold(&s.hold),
    }
}

/// Export `def` as a [`SourceScenario`].
///
/// `topology` and `duration` come from the test's `#[ktstr_scenario]`
/// attributes (via its `KtstrTestEntry`) rather than from the `ScenarioDef`,
/// because that is where they live — the scenario value carries the workload,
/// the attributes carry the machine it runs on.
///
/// `default_workers_per_cgroup` is recorded explicitly. A `WorkSpec` with
/// `num_workers: None` means "inherit", and the two backends would otherwise
/// inherit *different* defaults — which is precisely the kind of divergence
/// that makes a cross-backend comparison meaningless.
pub fn export_scenario(
    name: &str,
    def: &ScenarioDef,
    topology: &Topology,
    duration: std::time::Duration,
    default_workers_per_cgroup: u32,
) -> Export {
    let mut gaps = Vec::new();
    let steps: Vec<SourceStep> = def
        .steps()
        .iter()
        .enumerate()
        .map(|(i, s)| step(s, i, &mut gaps))
        .collect();

    if def.checks().is_some() {
        gaps.push(ExportGap {
            where_: "scenario".to_string(),
            construct: "Assert override".to_string(),
            reason: "the record carries the workload, not the oracle; a shared \
                     oracle has to be expressed against both backends' outputs, \
                     not smuggled through the workload record"
                .to_string(),
        });
    }

    let record = SourceScenario {
        name: name.to_string(),
        topology: SourceTopology {
            numa_nodes: topology.numa_nodes,
            llcs: topology.llcs,
            cores: topology.cores_per_llc,
            threads: topology.threads_per_core,
        },
        duration: DurationNs::from_std(duration),
        steps,
        default_workers_per_cgroup,
    };

    Export { record, gaps }
}

#[cfg(test)]
mod tests;
