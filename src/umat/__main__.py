from dataclasses import dataclass

import cappa

from .conf import (
    AddLabelConf,
    AssignConf,
    BoundaryConf,
    DistributedSegConf,
    PreviewConf,
    RetrainConf,
    SampleConf,
    SpotConf,
)


@cappa.command(name="umat")
@dataclass
class Umat:
    command: cappa.Subcommands[
        AddLabelConf | AssignConf | BoundaryConf | PreviewConf | RetrainConf | SampleConf | DistributedSegConf | SpotConf
    ]


def main():

    conf = cappa.parse(Umat, completion=False)
    print(f"config: {conf.command}", flush=True)

    match conf.command:
        case AddLabelConf():
            from .tools.addlab import run
        case AssignConf():
            from .tools.assign import run
        case BoundaryConf():
            from .tools.boundary import run
        case PreviewConf():
            from .tools.preview import run
        case RetrainConf():
            from .tools.retrain import run
        case SampleConf():
            from .tools.sample import run
        case DistributedSegConf():
            from .tools.retrain import run
        case SpotConf():
            from .tools.spot import run

    run(conf.command)  # pyright: ignore


if __name__ == "__main__":
    main()
