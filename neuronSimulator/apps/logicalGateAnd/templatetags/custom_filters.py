from django import template

register = template.Library()

@register.filter
def index(sequence, position):
    """Retorna el elemento en la posición dada de una lista."""
    try:
        return sequence[position]
    except (IndexError, TypeError):
        return None
