/* sensation.csv */
select text, label, document, year, annotator
from (
	select json_extract(content, '$.text') as text, json_extract(content, '$.answer') as answer, json_extract(content, '$.accept[0]') as label, 
	json_extract(json_extract(content, '$.meta'), "$.id") as document, substring(json_extract(json_extract(content, '$.meta'), "$.date"), 1, 4) as year,
	substring(json_extract(content, '$._annotator_id'), 25) as annotator
	from example
	where answer == 'accept' and label is not null
)
where annotator in (
	select annotator
	from (
		select json_extract(content, '$.text') as text, json_extract(content, '$.answer') as answer, json_extract(content, '$.accept[0]') as label, 
		json_extract(json_extract(content, '$.meta'), "$.id") as document, substring(json_extract(json_extract(content, '$.meta'), "$.date"), 1, 4) as year,
		substring(json_extract(content, '$._annotator_id'), 25) as annotator
		from example
		where answer == 'accept' and label is not null
	)
	group by annotator
	having count(annotator) > 10
)