/* 
select json_extract(content, '$.answer') as answer, json_extract(content, '$.accept[0]') as accept, count(1)
from example
where answer == 'accept' and accept is not null
group by 1,2 
*/


/* sensation_short.csv */
/*
select text, accept
from (
	select json_extract(content, '$.text') as text, json_extract(content, '$.answer') as answer, json_extract(content, '$.accept[0]') as accept
	from example
	where answer == 'accept' and accept is not null
)
*/

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

/* senstaion_all_columns.csv */
/*
select json_extract(content, '$.text') as text, json_extract(content, '$.tokens') as tokens,
json_extract(json_extract(content, '$.meta'), "$.id") as document,
substring(json_extract(json_extract(content, '$.meta'), "$.date"), 1, 4) as year, substring(json_extract(json_extract(content, '$.meta'), "$.date"), 6, 2) as month,
substring(json_extract(json_extract(content, '$.meta'), "$.date"), 9, 2) as day,
json_extract(content, '$.accept[0]') as label, json_extract(content, '$.answer') as answer, 
substring(json_extract(content, '$._annotator_id'), 25) as annotator
from example
*/