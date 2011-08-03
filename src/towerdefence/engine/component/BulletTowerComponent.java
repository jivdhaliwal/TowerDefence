package towerdefence.engine.component;

import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.geom.Rectangle;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;

import towerdefence.GameplayState;
import towerdefence.engine.ResourceManager;
import towerdefence.engine.Settings;
import towerdefence.engine.entity.Bullet;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Entity;
import towerdefence.engine.entity.Tower;

public class BulletTowerComponent extends RenderComponent{
	
	Bullet bullet;
	Tower tower;
	Critter targetCritter;
	Critter tempTarget;
	
	int shootingCounter;
	
	float turretRotation=-90f;
	
	private final int waitingTime;
	
	private final float bulletSpeed;
	
	private final boolean isRocket;
	
	public BulletTowerComponent(String id, int waitingTime, float bulletSpeed) {
		super(id);
		bullet = new Bullet("Bullet");
		bullet.AddComponent(new ImageRenderComponent("ImageRenderComponent", 
				ResourceManager.getInstance().getImage("BULLET")));
		bullet.setCollisionBlock(new Rectangle(0, 0, 10, 10));
		
		targetCritter=null;
		
		this.waitingTime = waitingTime;
		this.bulletSpeed = bulletSpeed;
		isRocket = false;
	}
	
	public BulletTowerComponent(String id, int waitingTime, float bulletSpeed, boolean isRocket) {
		super(id);
		bullet = new Bullet("Bullet");
		bullet.AddComponent(new ImageRenderComponent("ImageRenderComponent", 
				ResourceManager.getInstance().getImage("BULLET")));
		bullet.setCollisionBlock(new Rectangle(0, 0, 7, 7));
		
		targetCritter=null;
		
		this.waitingTime = waitingTime;
		this.bulletSpeed = bulletSpeed;
		this.isRocket = isRocket;
	}
	
	@Override
    public void setOwnerEntity(Entity entity)
    {
        this.entity = entity;
        tower = (Tower)entity;
        bullet.setPosition(tower.getPosition());
        bullet.getCollisionBlock().setLocation(bullet.getPosition());
    }
	
	@Override
	public void update(GameContainer gc, StateBasedGame sb, int delta) {
		
		shootingCounter-=delta;
		
		targetCritter = tower.getTargetCritter();
		
		if(tempTarget==null) {
			tempTarget=targetCritter;
		}
		
		if(tempTarget!=null && bullet!=null) {
			
			Vector2f centreCritter = new Vector2f( tempTarget.getPosition().x+(GameplayState.TILESIZE/2), 
					tempTarget.getPosition().y+(GameplayState.TILESIZE/2));
			
			Vector2f direction = (centreCritter.sub(bullet.getPosition()));
			direction.normalise();
			bullet.setPosition(bullet.getPosition().add((direction.scale(bulletSpeed))));
			
			if(bullet.getCollisionBlock().intersects(tempTarget.getCollisionBlock())) {
				tempTarget.takeDamage(Settings.getInstance().getBaseDPS()[tower.getType()]);
				bullet=null;
				tempTarget=null;
				shootingCounter=waitingTime;
			}
			
			
		} else if(bullet==null && shootingCounter<0) {
			if(tempTarget!=null) {
				createBullet();
			}
			
		}
		
		if(bullet!=null) {
			bullet.update(gc, sb, delta);
		}

		
	}

	@Override
	public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
		
		if(tempTarget!=null) {
			 turretRotation = (float)tempTarget.getPosition().sub(tower.getPosition()).getTheta();
		}
			
		// Draw tower's turret (which will rotate towards critters)
		gr.rotate(tower.getPosition().x + 16, tower.getPosition().y + 16,
				turretRotation + 90);
		if(isRocket) {
			ResourceManager.getInstance().getImage("ROCKET_TURRET")
			.draw(tower.getPosition().x, tower.getPosition().y);
		} else {
			ResourceManager.getInstance().getImage("BULLET_TURRET")
					.draw(tower.getPosition().x, tower.getPosition().y);
		}
		gr.rotate(tower.getPosition().x + 16, tower.getPosition().y + 16,
				(float) -(turretRotation) - 90);
		
		if(bullet!=null && tempTarget!=null) {
			bullet.render(gc, sb, gr);
		}
		
	}
	
	public void createBullet() {
		bullet = new Bullet("Bullet");
		bullet.AddComponent(new ImageRenderComponent("ImageRenderComponent",
				ResourceManager.getInstance().getImage("BULLET")));
		bullet.setCollisionBlock(new Rectangle(tower.getPosition().x+16,tower.getPosition().y+16, 10, 10));
		bullet.setPosition(new Vector2f(tower.getPosition().x+16,tower.getPosition().y+16));
	}

}
