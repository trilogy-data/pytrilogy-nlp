

### New Test Case

Go [here](https://github.com/duckdb/duckdb/tree/main/extension/tpcds/dsdgen/queries).

Copy SQL to query.sql.

Use that to create query.preql and query .prompt.

Prompt format should specify expected inputs, the attempts, and the target success rate.

Prompts are .toml files.

ex:

[Easy]
attempts = 4
target = 0.75
imports = ["store_sales"]
prompt = """Using just store_sales data, return a rows with the fields item name, average quantity sold, average list price, average coupon amount, and average sales price
 for male customers who are single with College education in the year 2000 and did not come from the event OR email promotion channels. (hint - have an 'OR' group checking for 'N' for both of those).

 Use the demographic inputs, use the point in time value from customer sale.

Order by the item name asc"""