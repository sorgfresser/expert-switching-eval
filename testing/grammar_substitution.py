import re


def specific_attributes_to_grammar(grammar_lines: list[str], attribute_names: list[str]) -> list[str]:
    """
    Replace the attribute line in an EBNF grammar with a new line that includes only the specified attributes.

    :param grammar_lines: The lines of the EBNF grammar
    :param attribute_names: The names of the attributes to include
    :return: The new grammar lines
    """
    grammar_lines = [
        line.strip() for line in grammar_lines if not line.strip().startswith("#")
    ]
    # Find the attribute line
    attribute_lines = [line for line in grammar_lines if line.startswith("single_attribute ::=")]
    assert len(attribute_lines) == 1, "There should be exactly one attribute line in the grammar"

    # Add surrounding quotes to attribute names
    attribute_names = [f'"{attribute_name.replace("\"", "'")}"' for attribute_name in attribute_names]
    attribute_idx = grammar_lines.index(attribute_lines[0])

    new_attribute_line = f"single_attribute ::= {' | '.join(attribute_names)}"
    grammar_lines[attribute_idx] = new_attribute_line
    return grammar_lines


def specific_tables_to_grammar(grammar_lines: list[str], table_names: list[str]) -> list[str]:
    """
    Replace the table line in an EBNF grammar with a new line that includes only the specified tables.

    :param grammar_lines: The lines of the EBNF grammar
    :param table_names: The names of the tables to include
    :return: The new grammar lines
    """
    grammar_lines = [
        line.strip() for line in grammar_lines if not line.strip().startswith("#")
    ]
    # Find the table line
    table_lines = [line for line in grammar_lines if line.startswith("table ::=")]
    assert len(table_lines) == 1, "There should be exactly one table line in the grammar"

    # Add surrounding quotes to table names
    table_names = [f'"{table_name.replace("\"", "'")}"' for table_name in table_names]
    table_idx = grammar_lines.index(table_lines[0])

    new_table_line = f"table ::= {' | '.join(table_names)}"
    grammar_lines[table_idx] = new_table_line
    return grammar_lines


def tables_from_sql(sql: str) -> list[str]:
    """
    Extract the table names from a SQL CREATE TABLE query.

    :param sql: The SQL query
    :return: The table names
    """
    # Remove comments
    sql = re.sub(r'--.*\n', '', sql)
    # Extract table names
    table_names = re.findall(r'CREATE TABLE (\w+)', sql)
    return table_names


def attributes_from_sql(sql: str) -> list[str]:
    """
    Extract the attribute names from a SQL CREATE TABLE query.

    :param sql: The SQL query
    :return: The attribute names
    """
    attributes = []
    # Remove comments
    sql = re.sub(r'--.*\n', '', sql)
    # Extract attribute names
    attribute_blocks = re.findall(r'\([^;]*\);', sql, re.DOTALL)
    for attribute_block in attribute_blocks:
        attribute_block = attribute_block.strip().removeprefix("(").removesuffix(");").strip()
        attribute_statements = attribute_block.split(",")
        # Delete foreign key, primary key, and unique constraints
        attribute_statements = [statement for statement in attribute_statements if
                                not statement.strip().startswith("FOREIGN KEY") and not statement.strip().startswith(
                                    "PRIMARY KEY") and not statement.strip().startswith("UNIQUE")]

        attribute_statements = [statement.strip() for statement in attribute_statements]
        attribute_names = [statement.split()[0] for statement in attribute_statements]
        attributes.extend(attribute_names)
    return list(set(attributes))


def substitute_in_grammar(
        grammar_lines: list[str],
        create_sql: str,
):
    """
    Substitute the tables and attributes from a SQL CREATE TABLE query into the grammar.

    :param grammar_lines: Grammar lines
    :param create_sql: SQL CREATE TABLE query
    :return: New grammar lines
    """
    table_names = tables_from_sql(create_sql)
    attribute_names = attributes_from_sql(create_sql)
    grammar_lines = specific_tables_to_grammar(grammar_lines, table_names)
    grammar_lines = specific_attributes_to_grammar(grammar_lines, attribute_names)
    return grammar_lines
