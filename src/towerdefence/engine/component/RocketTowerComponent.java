package towerdefence.engine.component;

import org.newdawn.slick.geom.Rectangle;
import org.newdawn.slick.geom.Vector2f;

import towerdefence.engine.ResourceManager;
import towerdefence.engine.entity.Bullet;

public class RocketTowerComponent extends ShootingTowerComponent {

	public RocketTowerComponent(String id, float bulletSpeed) {
		super(id);
		this.bulletSpeed = bulletSpeed;
		turretString="ROCKET_TURRET";
	}
	
	@Override
	public void createBullet() {
		bullet = new Bullet("Bullet");
		bullet.AddComponent(new ImageRenderComponent("ImageRenderComponent",
				ResourceManager.getInstance().getImage("BULLET")));
		bullet.setCollisionBlock(new Rectangle(tower.getPosition().x+11,tower.getPosition().y+16, 10, 10));
		bullet.setPosition(new Vector2f(tower.getPosition().x+11,tower.getPosition().y+16));
	}

}
