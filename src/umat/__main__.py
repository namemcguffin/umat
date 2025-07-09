from dataclasses import dataclass

import cappa

from .conf import (
    AddLabelConf,
    AssignConf,
    BoundaryConf,
    DistributedSegConf,
    FromProsegConf,
    PreviewConf,
    RetrainConf,
    SampleConf,
    SignalsConf,
    SpotConf,
)


@cappa.command(name="umat")
@dataclass
class Umat:
    command: cappa.Subcommands[
        AddLabelConf
        | AssignConf
        | BoundaryConf
        | DistributedSegConf
        | FromProsegConf
        | PreviewConf
        | RetrainConf
        | SampleConf
        | SignalsConf
        | SpotConf
    ]


def main():

    conf = cappa.parse(Umat, completion=False)
    print(f"config: {conf.command}", flush=True)

    match conf.command:
        case AddLabelConf():
            from .tools.addlab import run  # fmt: skip
            run(conf.command)
        case AssignConf():
            from .tools.assign import run  # fmt: skip
            run(conf.command)
        case BoundaryConf():
            from .tools.boundary import run  # fmt: skip
            run(conf.command)
        case DistributedSegConf():
            from .tools.segd import run  # fmt: skip
            run(conf.command)
        case FromProsegConf():
            from .tools.from_proseg import run  # fmt: skip
            run(conf.command)
        case PreviewConf():
            from .tools.preview import run  # fmt: skip
            run(conf.command)
        case RetrainConf():
            from .tools.retrain import run  # fmt: skip
            run(conf.command)
        case SampleConf():
            from .tools.sample import run  # fmt: skip
            run(conf.command)
        case SignalsConf():
            from .tools.signals import run  # fmt: skip
            run(conf.command)
        case SpotConf():
            from .tools.spot import run  # fmt: skip
            run(conf.command)


if __name__ == "__main__":
    main()
