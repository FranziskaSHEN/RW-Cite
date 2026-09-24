import argparse
import re


def main():
    args = parse_command_line()
    data = read(args.input)
    data = convert(data)
    write(args.output, data)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description='Replace \\def with \\newcommand where possible.',
    )
    parser.add_argument(
        'input',
        help='TeX input file with \\def',
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='TeX output file with \\newcommand',
    )

    return parser.parse_args()

def read(path):
    with open(path, mode='rb') as handle:
        return handle.read()


def convert(data):
    return re.sub(
        rb'((?:\\(?:expandafter|global|long|outer|protected)'
        rb'(?: +|\r?\n *)?)*)?'
        rb'\\def *(\\[a-zA-Z]+) *(?:#+([0-9]))*\{',
        replace,
        data,
    )


def replace(match):
    prefix = match.group(1)
    if (
            prefix is not None and
            (
                b'expandafter' in prefix or
                b'global' in prefix or
                b'outer' in prefix or
                b'protected' in prefix
            )
    ):
        pass #return match.group(0)

    result = rb'\newcommand'

    result += b'{' + match.group(2) + b'}'
    if match.lastindex == 3:
        result += b'[' + match.group(3) + b']'

    result += b'{'
    return result


def write(path, data):
    with open(path, mode='wb') as handle:
        handle.write(data)

    print('=> File written: {0}'.format(path))


if __name__ == '__main__':
    main()