using System;
using System.Collections.Generic;
using CPORLib.LogicalUtilities;

namespace CPORLib.PlanningModel
{
    public class ParametrizedAction : PlanningAction
    {
        public List<Parameter> Parameters { get; private set; }
        public ParametrizedAction(string sName)
            : base(sName)
        {
            Parameters = new List<Parameter>();
            ParameterNameToType = new Dictionary<string, string>();
        }
        public Dictionary<string, string> ParameterNameToType { get; private set; }
        public void AddParameter(Parameter parameter)
        {
            Parameters.Add(parameter);
            ParameterNameToType[parameter.Name] = parameter.Type;
        }
        public void AddParameter(string sName, string sType)
        {
            Parameter parameter = new Parameter(sType, sName);
            AddParameter(parameter);
        }
        public override PlanningAction Clone()
        {
            ParametrizedAction aNew = new ParametrizedAction(Name);
            aNew.Parameters = Parameters;
            if (Preconditions != null)
                aNew.Preconditions = Preconditions.Clone();
            if (Effects != null)
                aNew.SetEffects(Effects.Clone());
            if (Observe != null)
                aNew.Observe = Observe.Clone();
            aNew.HasConditionalEffects = HasConditionalEffects;
            aNew.ContainsNonDeterministicEffect = ContainsNonDeterministicEffect;
            return aNew;
        }

        public override string ToString()
        {
            string s = "(:action " + Name + "\n";
            s += " :parameters (";
            foreach (Parameter p in Parameters)
            {
                s += p.Name + " - " + p.Type + " ";
            }
            s += ")\n";
            if (Preconditions != null)
                s += " :precondition " + Preconditions + "\n";
            if (Effects != null)
                s += " :effect " + Effects + "\n";
            if (Observe != null)
                s += " :observe " + Observe + "\n";
            s += ")";
            return s;
        }

        private void FixParametersNames(ParametrizedPredicate pp)
        {
            foreach (Argument a in pp.Parameters)
            {
                if (a is Parameter param)
                {
                    if (!param.Name.StartsWith("?"))
                        param.Name = "?" + param.Name;
                }
            }
        }
        private void FixParametersNames(Formula f)
        {
            if (f == null)
                return;
            // Walk the formula tree directly so every predicate instance is touched,
            // even if multiple identical predicates appear in different branches.
            if (f is PredicateFormula pf)
            {
                if (pf.Predicate is ParametrizedPredicate pp)
                    FixParametersNames(pp);
            }
            else if (f is CompoundFormula cf)
            {
                foreach (Formula sub in cf.Operands)
                    FixParametersNames(sub);
            }
            else if (f is ProbabilisticFormula prob)
            {
                foreach (Formula sub in prob.Options)
                    FixParametersNames(sub);
            }
        }

        public void FixParametersNames()
        {
            foreach(Parameter p in Parameters)
            {
                if(!p.Name.StartsWith("?"))
                    p.Name = "?" + p.Name;
            }
            FixParametersNames(Preconditions);
            FixParametersNames(Effects);
            FixParametersNames(Observe);

        }
    }
}
