from dataclasses import dataclass

import cappa

from . import conf as c


@cappa.command(name="umat")
@dataclass
class Umat:
    command: cappa.Subcommands[
        c.AddLabelConf
        | c.AssignConf
        | c.BoundaryConf
        | c.DistributedSegConf
        | c.FromProsegConf
        | c.PreviewConf
        | c.RetrainConf
        | c.SampleConf
        | c.SignalsConf
        | c.SpotConf
    ]


def main():
    conf = cappa.parse(Umat, completion=False)
    print(f"config: {conf.command}", flush=True)

    # fmt: off
    match conf.command:
        case c.AddLabelConf():
            from .tools.addlab import run
            run(conf.command)
        case c.AssignConf():
            from .tools.assign import run
            run(conf.command)
        case c.BoundaryConf():
            from .tools.boundary import run
            run(conf.command)
        case c.DistributedSegConf():
            from .tools.segd import run
            run(conf.command)
        case c.FromProsegConf():
            from .tools.from_proseg import run
            run(conf.command)
        case c.PreviewConf():
            from .tools.preview import run
            run(conf.command)
        case c.RetrainConf():
            from .tools.retrain import run
            run(conf.command)
        case c.SampleConf():
            from .tools.sample import run
            run(conf.command)
        case c.SignalsConf():
            from .tools.signals import run
            run(conf.command)
        case c.SpotConf():
            from .tools.spot import run
            run(conf.command)
    # fmt:on


if __name__ == "__main__":
    main()
