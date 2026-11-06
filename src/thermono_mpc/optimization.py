"""Explicitly configured SLSQP comparators for controlled scaling studies."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .controllers import MPCConfig, MPCResult, ROMPredictor, limit_power
from .diagnostics import plan_objective


class ConfiguredSLSQPMPC:
    """Keep the same physical problem/candidate set while varying coordinates.

The legacy SLSQPMPC remains unchanged. Scaling changes numerical coordinates,
not physical cost weights, actuator bounds, thermal limits or admissible plans.
"""
    def __init__(self, config, settings=MPCConfig(), *, normalized=True, starts=2,
                 max_iterations=80, ftol=1e-9, physical_fd_step_W=1.0):
        if starts not in (1,2):
            raise ValueError("starts must be one or two")
        self.process, self.settings = config, settings
        self.predictor = ROMPredictor(config)
        self.normalized, self.starts = bool(normalized), starts
        self.max_iterations, self.ftol, self.physical_fd_step_W = max_iterations, ftol, physical_fd_step_W
        self.warm = None
        self.local_calls = []
        self.selection = 0

    def optimize(self, field_K, zones_K, reference_K, previous_W, dt_s):
        s,c = self.settings,self.process
        ref = np.asarray(reference_K,dtype=float)
        if ref.shape != (s.horizon,) or not np.isfinite(ref).all():
            raise ValueError("reference shape mismatch")
        index = np.minimum(np.arange(s.horizon)*s.blocks//s.horizon,s.blocks-1)
        scale = np.tile(c.pmax,s.blocks).astype(float) if self.normalized else np.ones(s.blocks*c.zones)
        def sequence(v):
            return (v*scale).reshape(s.blocks,c.zones)[index]
        def predict(v):
            plan=sequence(v)
            p=self.predictor.predict(field_K,zones_K,plan[None],dt_s)
            return plan,p
        def objective(v):
            plan,p=predict(v)
            return float(plan_objective(p.temperature_K,plan[None],ref,previous_W,s,dt_s)[0])
        def slew_margin(v):
            blocks=(v*scale).reshape(s.blocks,c.zones)
            delta=np.diff(np.vstack([previous_W,blocks]),axis=0)
            return np.r_[(np.asarray(c.slew)-delta).ravel(),(np.asarray(c.slew)+delta).ravel()]
        def temperature_margin(v):
            _,p=predict(v)
            return np.r_[c.solid_limit-p.temperature_K[0].max(axis=(1,2)),c.zone_limit-p.zone_temperature_K[0].max(axis=1)]
        if self.warm is None:
            guess=np.broadcast_to(np.minimum(previous_W+c.slew,c.pmax),(s.blocks,c.zones)).copy()
        else:
            guess=self.warm[np.minimum(np.arange(s.blocks)*s.horizon//s.blocks+1,s.horizon-1)].copy()
        for j in range(s.blocks):
            guess[j]=limit_power(guess[j],previous_W if j==0 else guess[j-1],c)
        high=np.empty_like(guess);prev=np.asarray(previous_W).copy()
        for j in range(s.blocks):
            high[j]=limit_power(np.asarray(c.pmax),prev,c);prev=high[j]
        # Fixed acceptance candidates in every arm, including one-start arms.
        candidates=[guess.ravel()/scale,high.ravel()/scale,np.zeros(s.blocks*c.zones)]
        bounds=np.tile(c.pmax,s.blocks)/scale
        for local_index,start in enumerate((guess.ravel()/scale,high.ravel()/scale)[:self.starts]):
            initial_cost=objective(start)
            result=minimize(objective,start,method="SLSQP",bounds=list(zip(np.zeros_like(bounds),bounds)),
                constraints=[{"type":"ineq","fun":slew_margin},{"type":"ineq","fun":temperature_margin}],
                options={"maxiter":self.max_iterations,"ftol":self.ftol,
                         "eps":self.physical_fd_step_W/scale,"disp":False})
            candidates.append(result.x)
            physical=result.x*scale
            finite=np.isfinite(physical).all()
            self.local_calls.append({"selection":self.selection,"start":local_index,"normalized":self.normalized,
                "nit":int(result.nit),"nfev":int(result.nfev),"success":bool(result.success),"message":str(result.message),
                "initial_cost":initial_cost,"final_cost":float(result.fun),
                "minimum_slew_margin_W":float(slew_margin(result.x).min()) if finite else None,
                "minimum_thermal_margin_K":float(temperature_margin(result.x).min()) if finite else None,
                "physical_bound_violation_W":float(max(0.,(-physical).max(),(physical-np.tile(c.pmax,s.blocks)).max())) if finite else None})
        feasible=[]
        for v in candidates:
            physical=v*scale
            if (np.isfinite(physical).all() and np.all(physical>=-1e-5) and
                    np.all(physical<=np.tile(c.pmax,s.blocks)+1e-5) and
                    np.all(slew_margin(v)>=-1e-5) and np.all(temperature_margin(v)>=-1e-5)):
                value=objective(v)
                if np.isfinite(value):
                    feasible.append((value,sequence(v).copy()))
        self.selection+=1
        if not feasible:
            return MPCResult(np.full(c.zones,np.nan),np.empty((0,c.zones)),np.inf,0,"INFEASIBLE_CANDIDATES")
        value,plan=min(feasible,key=lambda x:x[0])
        self.warm=plan
        return MPCResult(plan[0],plan,value,len(feasible),"OK")
