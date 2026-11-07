-- name: Late rate by month, with the change from the previous month
WITH delivered AS (
    SELECT substr(order_purchase_timestamp, 1, 7) AS month,
           date(order_delivered_customer_date) > date(order_estimated_delivery_date) AS is_late
    FROM orders
    WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
),
monthly AS (
    SELECT month, COUNT(*) AS orders, ROUND(100.0 * AVG(is_late), 1) AS late_pct
    FROM delivered
    GROUP BY month
    HAVING COUNT(*) >= 500
)
SELECT month, orders, late_pct,
       ROUND(late_pct - LAG(late_pct) OVER (ORDER BY month), 1) AS change_pct_points
FROM monthly
ORDER BY month;

-- name: Worst seller-to-customer routes with at least 200 orders
SELECT s.seller_state || ' > ' || c.customer_state AS route,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(100.0 * COUNT(DISTINCT CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
                                         THEN o.order_id END) / COUNT(DISTINCT o.order_id), 1) AS late_pct,
       ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)), 1) AS avg_days
FROM orders o
JOIN order_items i ON i.order_id = o.order_id
JOIN sellers s ON s.seller_id = i.seller_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
GROUP BY route
HAVING orders >= 200
ORDER BY late_pct DESC
LIMIT 10;

-- name: Busiest sellers with the highest late rates, ranked with a window function
WITH seller_stats AS (
    SELECT i.seller_id,
           COUNT(DISTINCT o.order_id) AS orders,
           ROUND(100.0 * COUNT(DISTINCT CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
                                             THEN o.order_id END) / COUNT(DISTINCT o.order_id), 1) AS late_pct
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY i.seller_id
    HAVING orders >= 200
)
SELECT RANK() OVER (ORDER BY late_pct DESC) AS rank_by_late_rate,
       substr(seller_id, 1, 8) AS seller, orders, late_pct
FROM seller_stats
ORDER BY rank_by_late_rate
LIMIT 10;
