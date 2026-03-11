import numpy


PERSONA = 'You are a helpful assistant tasked with discovering mathematical function structures for scientific systems.'
PERSONA_ANALYSIS = 'You are an expert in {}.'


def _write_xml_prompt(role, context, task):
    prompt = '<prompt>\n'
    prompt += '<role>\n{}\n</role>\n'.format(role)
    prompt += '<context>\n{}\n</context>\n'.format(context)
    prompt += '<task>\n{}\n</task>\n'.format(task)
    prompt += '</prompt>'

    return prompt


def write_init_prompt(instruction):
    context = instruction
    task = 'Complete the equation function, considering the physical meaning and relationships between inputs variables.'

    return _write_xml_prompt(PERSONA, context, task)


def write_eq_analysis_prompt(domain, ref_code):
    context = '```Python\n'
    context += ref_code + '\n'
    context += '```'

    task = 'Analyze the relationships between the input and output variables of the equation function, and explain them from the perspective of {}.\n'.format(domain)
    task += 'Summarize your analysis results into compact statements.'

    return _write_xml_prompt(PERSONA_ANALYSIS.format(domain), context, task)


def write_guided_analysis_prompt(y_pred, dataset, code):
    errors = (dataset.y - y_pred)**2
    mse_oe = numpy.sum(errors[dataset.y <= y_pred])
    mse_ue = numpy.sum(errors[dataset.y > y_pred])

    role = PERSONA_ANALYSIS + '\n'
    if mse_oe >= mse_ue:
        role += 'Your goal is to suggest propositions to fix {} of a given equation function.'.format('overestimations')
    else:
        role += 'Your goal is to suggest propositions to fix {} of a given equation function.'.format('underestimations')

    context = '```Python\n'
    context += code + '\n'
    context += '```'

    if mse_oe >= mse_ue:
        task = 'Analyze the equation function and suggest your propositions to fix the {}.'.format('overestimation')
    else:
        task = 'Analyze the equation function and suggest your propositions to fix the {}.'.format('underestimation')

    return _write_xml_prompt(role, context, task)


def write_guided_update_prompt(y_pred, dataset, code, analysis_global, code_global):
    errors = (dataset.y - y_pred)**2
    mse_oe = numpy.sum(errors[dataset.y <= y_pred])
    mse_ue = numpy.sum(errors[dataset.y > y_pred])

    context = ('<message>Below is some prior knowledge about a function showing relatively low errors. '
               'You can leverage when updating the current equation function.</message>\n')
    context += '<prior_knowledge>\n'
    context += '<analysis>\n{}\n</analysis>\n'.format(analysis_global)
    context += '<function>\n```Python\n{}\n```\n</function>\n'.format(code_global)
    context += '</prior_knowledge>'

    if mse_oe >= mse_ue:
        task = '<message>Update the current function below to fix the {}.</message>\n'.format('overestimation')
    else:
        task = '<message>Update the current function below to fix the {}.</message>\n'.format('underestimation')
    task += '<current_function>\n```Python\n{}\n```\n</current_function>'.format(code)

    return _write_xml_prompt(PERSONA, context, task)
