import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wiki_dir', type=str, default='', help='directory to extracted wiki corpus')
parser.add_argument('--book_dir', type=str, default='', help='directory to extracted book corpus')
parser.add_argument('--output_dir', type=str, default='', help='txt path to formatted wiki corpus')
args = parser.parse_args()


# Put one article per line
def format_wiki_corpus():
    output_path = os.path.join(args.output_dir, 'wikicorpus_en_format.txt')
    with open(output_path, mode='w', newline='\n') as ofile:
        for dirname in glob.glob(args.wiki_dir + '/*/', recursive=False):
            for filename in glob.glob(dirname + 'wiki_*', recursive=True):
                print(filename)
                article_lines = []
                article_open = False

                with open(filename, mode='r', newline='\n') as file:
                    for line in file:
                        if '<doc id=' in line:
                            article_open = True
                        elif '</doc>' in line:
                            article_open = False
                            for oline in article_lines[1:]:
                                if oline != '\n':
                                    ofile.write(oline.rstrip() + ' ')
                            ofile.write('\n\n')
                            article_lines = []
                        else:
                            if article_open:
                                article_lines.append(line)


def format_book_corpus():
    output_path = os.path.join(args.output_dir, 'bookcorpus_format.txt')
    with open(output_path, mode='w', newline='\n') as ofile:
        for filename in glob.glob(args.book_dir + '/' + '*.txt', recursive=True):
            with open(filename, mode='r', encoding='utf-8-sig', newline='\n') as file:
                for i, line in enumerate(file):
                    if line.strip() != '':
                        ofile.write(line.strip() + ' ')
                    if (i + 1) % 20 == 0:
                        ofile.write("\n\n")
            ofile.write("\n\n")


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.wiki_dir:
        format_wiki_corpus()
    if args.book_dir:
        format_book_corpus()


if __name__ == '__main__':
    main()
