import click
from transformers import pipeline

# Hugging Face transformers
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

@click.command()
@click.option('-t', '--textfile', type=click.Path(exists=True), help='Path to the text file to summarize.')
@click.argument('text', required=False)
def main(textfile, text):
    if textfile:
        with open(textfile, 'r') as file:
            content = file.read()
        summary = summarize_text(content)
        click.echo(f"Summary of {textfile}:\n{summary}")
    elif text:
        summary = summarize_text(text)
        click.echo(f"Summary of provided text:\n{summary}")
    else:
        click.echo("Please provide either a text file with -t option or text as an argument.")

if __name__ == '__main__':
    main()
